import argparse
from tqdm import tqdm
from pydub import AudioSegment
import torchaudio
import os

def mp3_convert_wav(mp3_file, wav_file):
    try:
        sound = AudioSegment.from_mp3(mp3_file)
        sound=sound.set_frame_rate(16000)
        sound=sound.set_channels(1)
        sound=sound.set_sample_width(2)
        sound.export(wav_file, format="wav")
    except Exception as e:
        print(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--shard", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    args = parser.parse_args()

    assert args.rank < args.shard, f"rank: {args.rank} >= shard: {args.shard}"

    with open(args.input, 'r') as f:
        files = [line.strip() for line in f ]

    mp3_files = files[args.rank::args.shard]
    for mp3_file in tqdm(mp3_files):
        wav_file = mp3_file.replace("/clips/", "/wav/").replace(".mp3", ".wav")
        if os.path.exists(wav_file):
            try:
                torchaudio.info(wav_file)
            except Exception as e:
                print(e)
                mp3_convert_wav(mp3_file, wav_file)
        else:
            mp3_convert_wav(mp3_file, wav_file)

if __name__ == "__main__":
    main()
