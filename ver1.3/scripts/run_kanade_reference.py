from __future__ import annotations

import argparse

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Kanade voice-conditioned pseudo-reference wav.")
    parser.add_argument("--target-wav", required=True)
    parser.add_argument("--speaker-wav", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--model-id", default="frothywater/kanade-25hz-clean")
    args = parser.parse_args()

    import torch
    from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KanadeModel.from_pretrained(args.model_id).eval().to(device)
    vocoder = load_vocoder(model.config.vocoder_name).to(device)
    target_audio = load_audio(args.target_wav, sample_rate=model.config.sample_rate).to(device)
    speaker_audio = load_audio(args.speaker_wav, sample_rate=model.config.sample_rate).to(device)

    with torch.inference_mode():
        target_features = model.encode(target_audio)
        speaker_features = model.encode(speaker_audio)
        mel_spectrogram = model.decode(
            content_token_indices=target_features.content_token_indices,
            global_embedding=speaker_features.global_embedding,
        )
        waveform = vocode(vocoder, mel_spectrogram.unsqueeze(0))

    sf.write(
        args.out_wav,
        waveform.squeeze().detach().cpu().numpy(),
        int(model.config.sample_rate),
    )


if __name__ == "__main__":
    main()
