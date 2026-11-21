# Transcrição de áudios do WhatsApp (.opus)

Transcreve mensagens de voz do WhatsApp com a maior fidelidade possível, usando
o **Whisper large-v3** (modelo mais preciso da OpenAI, rodando localmente via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)). Nada é enviado
para serviços externos: o áudio é processado na própria máquina.

## Instalação

Requer Python 3.9+. O `ffmpeg` não é obrigatório (o faster-whisper decodifica
`.opus` sozinho).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na primeira execução o modelo (~3 GB) é baixado do Hugging Face.

## Uso

1. Exporte os áudios do WhatsApp (ex.: `PTT-20241121-WA0003.opus`) e coloque em `audios/`.
2. Rode:

```bash
python transcrever.py
```

Resultados em `transcricoes/`:

| Arquivo | Conteúdo |
|---|---|
| `<nome>.txt` | texto corrido |
| `<nome>.srt` | texto com marcação de tempo |
| `<nome>.json` | segmentos e palavras com tempo e probabilidade |
| `_todas_as_transcricoes.md` | todos os áudios em ordem cronológica, com palavras de baixa confiança destacadas para revisão |

### Opções úteis

```bash
# Nomes próprios / termos técnicos melhoram muito a grafia correta
python transcrever.py --contexto "Diogo, Lambda3, School of AI, scikit-learn"

# Máxima precisão numérica em CPU (mais lento)
python transcrever.py --precisao float32

# Se falas baixas estiverem sendo cortadas
python transcrever.py --sem-vad

# Arquivos/pastas específicos, outro idioma ou detecção automática
python transcrever.py ~/Downloads/conversa --idioma auto
```

## Por que estas configurações (fidelidade)

- **large-v3**: menor taxa de erro em português entre os modelos Whisper.
- **beam search (5) + fallback de temperatura**: decodificação cuidadosa; só
  "arrisca" quando o trecho sai repetitivo ou com baixa confiança.
- **`condition_on_previous_text=False`**: evita o problema conhecido do Whisper
  de repetir/inventar frases em áudios curtos ou com ruído.
- **VAD (detecção de voz)**: remove silêncios longos, que são a principal causa
  de alucinações ("Legendas por..." etc.).
- **Confiança por palavra**: o consolidado lista as palavras com probabilidade
  < 50% para revisão manual.

Com GPU NVIDIA o script usa CUDA automaticamente (bem mais rápido). Em CPU,
espere algo próximo da duração do áudio.
