import os
import json
import tempfile
import subprocess
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        """Setup: download and cache model weights"""
        os.makedirs("weights", exist_ok=True)
        print("🔽 Checking or downloading required model weights...")

        # Cache (only downloads once, stored in ~/.cache/huggingface)
        snapshot_download("MeiGen-AI/InfiniteTalk", local_dir="weights/InfiniteTalk", ignore_patterns=[".git*"])
        snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="weights/Wan2.1-I2V-14B-480P", ignore_patterns=[".git*"])
        snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir="weights/chinese-wav2vec2-base", ignore_patterns=[".git*"])

    def predict(
        self,
        image: Path = Input(description="Input image (jpg/png)"),
        audio: Path = Input(description="Input audio file (mp3/wav)"),
        resolution: str = Input(default="infinitetalk-480", choices=["infinitetalk-480", "infinitetalk-720"]),
        mode: str = Input(default="streaming", choices=["streaming", "clip"]),
        device: str = Input(default="cuda", choices=["cpu", "cuda"]),  # ✅ GPU default
    ) -> Path:
        """Run inference: generate 20s talking video from image + audio"""
        input_json = "input.json"
        with open(input_json, "w") as f:
            json.dump({
                "image_path": str(image),
                "audio_path": str(audio)
            }, f)

        os.makedirs("output", exist_ok=True)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = tmpfile.name

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
            "--max_frame_num", "500",  # ~20 seconds
            "--save_file", output_path,
        ]

        # ✅ Force CPU mode only when explicitly selected
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            cmd += ["--device", "cpu"]
        else:
            cmd += ["--device", "cuda"]

        print(f"🚀 Running InfiniteTalk generation on {device.upper()} ...")
        subprocess.run(cmd, check=True)

        print(f"✅ Done! Output saved at: {output_path}")
        return Path(output_path)
