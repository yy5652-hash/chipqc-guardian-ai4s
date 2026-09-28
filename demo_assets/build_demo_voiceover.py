"""Create voiceover tracks aligned to the seven captioned demo scenes."""

from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FFMPEG = ROOT / ".video-deps/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"
DURATIONS = [10, 10, 12, 13, 13, 14, 10]
NARRATION = [
    "ChipQC Guardian is a human-reviewed image quality workflow for organ-on-chip research.",
    "Brightfield images are inspected repeatedly. A model score alone cannot decide whether a sample is suitable.",
    "Our running prototype loads a research model, accepts images, and states its limits clearly. These demo inputs are synthetic.",
    "A synthetic checker image receives a rule preview of pass. This is not a validated decision for real samples.",
    "A flat synthetic image triggers a reacquire preview because clarity and contrast are low. The model can disagree, so people retain the final decision.",
    "On a frozen test split of twelve acquisition groups and six hundred seventy images, balanced accuracy is point seven one two and A U R O C is point seven seven two. Uncertainty remains wide.",
    "The public source package, split manifest, and technical report are available on GitHub. Next we need prospective and external-lab evaluation.",
]

segments = []
for index, (text, duration) in enumerate(zip(NARRATION, DURATIONS), 1):
    aiff = HERE / f"voice-{index:02d}.aiff"
    wav = HERE / f"voice-{index:02d}.wav"
    subprocess.run(["say", "-v", "Samantha", "-r", "165", "-o", str(aiff), text], check=True)
    subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff),
         "-af", f"apad,atrim=0:{duration}", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    segments.append(wav)

concat_file = HERE / "voice-concat.txt"
concat_file.write_text("".join(f"file '{p}'\n" for p in segments))
audio = HERE / "voiceover.wav"
subprocess.run(
    [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
     "-i", str(concat_file), "-c:a", "pcm_s16le", str(audio)],
    check=True,
)
print(f"Saved {audio}")
