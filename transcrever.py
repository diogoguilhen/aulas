#!/usr/bin/env python3
"""
Transcrição de áudios do WhatsApp (.opus / .ogg / .m4a / .mp3 / .wav) com
foco em fidelidade máxima, usando Whisper large-v3 via faster-whisper.

Uso:
    python transcrever.py                       # transcreve tudo em ./audios
    python transcrever.py caminho/arquivo.opus  # arquivos ou pastas específicos
    python transcrever.py --modelo large-v3 --idioma pt --saida transcricoes

Saídas (por áudio, em ./transcricoes):
    <nome>.txt   texto corrido
    <nome>.srt   legenda com marcação de tempo
    <nome>.json  segmentos + palavras com tempo e probabilidade
e um consolidado  _todas_as_transcricoes.md  em ordem cronológica.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

EXTENSOES = {".opus", ".ogg", ".oga", ".m4a", ".mp3", ".wav", ".aac", ".amr", ".mp4", ".webm"}

# Ex.: "PTT-20241121-WA0003.opus", "AUD-20240102-WA0010.opus",
#      "WhatsApp Audio 2024-11-21 at 14.03.22.opus"
RE_PTT = re.compile(r"(?:PTT|AUD)-(\d{8})-WA(\d+)", re.I)
RE_WA_DESKTOP = re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})", re.I)


def chave_ordem(caminho: Path):
    """Ordena pela data/sequência embutida no nome do WhatsApp, se existir."""
    nome = caminho.name
    m = RE_PTT.search(nome)
    if m:
        return (m.group(1), int(m.group(2)), nome)
    m = RE_WA_DESKTOP.search(nome)
    if m:
        data = m.group(1).replace("-", "")
        hora = int(m.group(2)) * 10000 + int(m.group(3)) * 100 + int(m.group(4))
        return (data, hora, nome)
    return ("99999999", 0, nome)


def descricao_data(caminho: Path) -> str:
    m = RE_PTT.search(caminho.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").strftime("%d/%m/%Y")
    m = RE_WA_DESKTOP.search(caminho.name)
    if m:
        d = datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%d/%m/%Y")
        return f"{d} {int(m.group(2)):02d}:{m.group(3)}:{m.group(4)}"
    return ""


def coletar(entradas: list[str]) -> list[Path]:
    arquivos: list[Path] = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            arquivos += [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSOES]
        elif p.is_file():
            arquivos.append(p)
        else:
            print(f"[aviso] não encontrado: {e}", file=sys.stderr)
    return sorted(set(arquivos), key=chave_ordem)


def ts_srt(seg: float) -> str:
    ms = int(round(seg * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_curto(seg: float) -> str:
    m, s = divmod(int(seg), 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcrição fiel de áudios do WhatsApp")
    ap.add_argument("entradas", nargs="*", default=["audios"], help="arquivos ou pastas (padrão: ./audios)")
    ap.add_argument("--saida", default="transcricoes", help="pasta de saída (padrão: ./transcricoes)")
    ap.add_argument("--modelo", default="large-v3",
                    help="modelo Whisper (padrão: large-v3, o mais preciso). Ex.: large-v2, medium, small")
    ap.add_argument("--idioma", default="pt", help="código do idioma (padrão: pt). Use 'auto' para detectar")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--precisao", default=None,
                    help="compute_type (padrão: float16 em GPU, int8 em CPU; use float32 p/ máxima precisão em CPU)")
    ap.add_argument("--beam", type=int, default=5, help="beam size (padrão: 5)")
    ap.add_argument("--contexto", default=None,
                    help="texto de contexto: nomes próprios, jargões, termos técnicos que aparecem nos áudios")
    ap.add_argument("--sem-vad", action="store_true",
                    help="desliga o filtro de voz (VAD). Use se trechos baixos estiverem sendo cortados")
    ap.add_argument("--refazer", action="store_true", help="retranscreve mesmo se a saída já existir")
    args = ap.parse_args()

    arquivos = coletar(args.entradas)
    if not arquivos:
        print("Nenhum áudio encontrado. Coloque os arquivos .opus na pasta ./audios", file=sys.stderr)
        return 1

    from faster_whisper import WhisperModel  # import tardio: --help funciona sem o pacote

    dispositivo = args.dispositivo
    if dispositivo == "auto":
        try:
            import ctranslate2
            dispositivo = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dispositivo = "cpu"
    precisao = args.precisao or ("float16" if dispositivo == "cuda" else "int8")

    print(f"Carregando modelo {args.modelo} ({dispositivo}/{precisao})...", flush=True)
    modelo = WhisperModel(args.modelo, device=dispositivo, compute_type=precisao)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    idioma = None if args.idioma == "auto" else args.idioma
    consolidado: list[str] = []

    for i, arq in enumerate(arquivos, 1):
        base = saida / arq.stem
        json_path = base.with_suffix(".json")
        print(f"[{i}/{len(arquivos)}] {arq.name}", flush=True)

        if json_path.exists() and not args.refazer:
            dados = json.loads(json_path.read_text(encoding="utf-8"))
            print("    já transcrito (use --refazer para refazer)")
        else:
            t0 = time.time()
            segmentos, info = modelo.transcribe(
                str(arq),
                language=idioma,
                task="transcribe",
                beam_size=args.beam,
                best_of=5,
                patience=1.0,
                # fallback de temperatura: tenta greedy determinístico e só
                # aumenta se o trecho sair repetitivo ou com baixa confiança
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                # False evita o "loop" de alucinação em que o Whisper repete
                # frases do trecho anterior — importante em áudios curtos/ruidosos
                condition_on_previous_text=False,
                initial_prompt=args.contexto,
                word_timestamps=True,
                vad_filter=not args.sem_vad,
                vad_parameters={"min_silence_duration_ms": 700, "speech_pad_ms": 400},
            )
            segs = []
            for s in segmentos:
                segs.append({
                    "inicio": round(s.start, 2),
                    "fim": round(s.end, 2),
                    "texto": s.text.strip(),
                    "confianca_media": round(s.avg_logprob, 3),
                    "prob_sem_fala": round(s.no_speech_prob, 3),
                    "palavras": [
                        {"p": w.word, "i": round(w.start, 2), "f": round(w.end, 2), "prob": round(w.probability, 3)}
                        for w in (s.words or [])
                    ],
                })
            dados = {
                "arquivo": arq.name,
                "data": descricao_data(arq),
                "idioma": info.language,
                "prob_idioma": round(info.language_probability, 3),
                "duracao_s": round(info.duration, 2),
                "modelo": args.modelo,
                "segmentos": segs,
            }
            json_path.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

            texto = " ".join(s["texto"] for s in segs).strip()
            base.with_suffix(".txt").write_text(texto + "\n", encoding="utf-8")
            with base.with_suffix(".srt").open("w", encoding="utf-8") as f:
                for n, s in enumerate(segs, 1):
                    f.write(f"{n}\n{ts_srt(s['inicio'])} --> {ts_srt(s['fim'])}\n{s['texto']}\n\n")
            print(f"    {info.duration:.0f}s de áudio em {time.time() - t0:.0f}s")

        # Palavras com baixa confiança ficam marcadas no consolidado para revisão humana
        incertas = [w["p"].strip() for s in dados["segmentos"] for w in s["palavras"] if w["prob"] < 0.5]
        cab = f"## {dados['arquivo']}"
        if dados.get("data"):
            cab += f" — {dados['data']}"
        cab += f" ({ts_curto(dados['duracao_s'])})"
        linhas = [cab, ""]
        for s in dados["segmentos"]:
            linhas.append(f"[{ts_curto(s['inicio'])}] {s['texto']}")
        if incertas:
            linhas += ["", f"> Revisar (baixa confiança): {', '.join(incertas)}"]
        consolidado.append("\n".join(linhas))

    (saida / "_todas_as_transcricoes.md").write_text(
        "# Transcrições\n\n" + "\n\n---\n\n".join(consolidado) + "\n", encoding="utf-8"
    )
    print(f"\nPronto. Resultados em: {saida.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
