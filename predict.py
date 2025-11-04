import os
import json
import subprocess
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        os.makedirs("weights", exist_ok=True)
        print("🔽 Checking or downloading required model weights...")

        # cache: sadece yoksa indir
        snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="weights/Wan2.1-I2V-14B-480P", ignore_patterns=[".git*"])
        snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir="weights/chinese-wav2vec2-base", ignore_patterns=[".git*"])
        snapshot_download("MeiGen-AI/InfiniteTalk", local_dir="weights/InfiniteTalk", ignore_patterns=[".git*"])

    def predict(
        self,
        image: Path = Input(description="Input image (jpg/png)"),
        audio: Path = Input(description="Input audio file (mp3/wav)"),
        resolution: str = Input(default="infinitetalk-480", description="Video resolution (480 or 720)"),
        mode: str = Input(default="streaming", description="Generation mode: streaming or clip"),
        device: str = Input(default="cpu", choices=["cpu", "cuda"], description="Run on CPU or GPU"),
    ) -> Path:
        input_json = "input.json"
        with open(input_json, "w") as f:
            json.dump({
                "image_path": str(image),
                "audio_path": str(audio)
            }, f)

        os.makedirs("output", exist_ok=True)
        output_file = "output/result.mp4"

        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", input_json,
            "--size", resolution,
            "--mode", mode,
            "--sample_steps", "40",
            "--motion_frame", "9",
            "--max_frame_num", "500",
            "--save_file", output_file,
        ]

        if device == "cpu":
            cmd += ["--device", "cpu"]

        print("🚀 Running InfiniteTalk generation...")
        subprocess.run(cmd, check=True)

        return Path(output_file)
