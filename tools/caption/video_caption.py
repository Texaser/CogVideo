import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/cogvlm2-llama3-caption"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
args = parser.parse_args([])


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def test():
    video_data = open('/mnt/bum/hanyi/data/video/327261452_screen_43.903_0.mp4', 'rb').read()
    temperature = 0.1

    # prompt = "What action are they doing for each player?"
    # prompt = "Please describe this video in detail including a play-byscoreboard-play narration that describes each significant action chronologically, capturing the flow of the game from start to finish. Elaborate on the key actions occurring in the game, such as how shots are made, how rebounds are secured, and any other pivotal moments, providing insights into the players' techniques, movements, and any challenges they face during these actions."
    # prompt = "Please provide the game detail"

    # prompt = "Describe this video in detail."
    # response1 = predict(prompt, video_data, temperature)
    # print(response1)

    # # prompt = "Please describe this video in detail following the time displayed on the scoreboard."
    # # prompt = "Provide a detailed description of every player's motion and timeclock"
    # # prompt = "For each key time interval visible on the scoreboard, "

    # prompt = "Please offer each player's motion"
    # response2 = predict(prompt, video_data, temperature)
    # print(response2)

    # prompt = "Please offer the detail of scoreboard"
    # response3 = predict(prompt, video_data, temperature)
    # print(response3)

    prompt = "Generate the video detail like this format: {At the 3rd quarter with 6 minutes and 42.6 seconds remaining, Cleveland Cavaliers’ players are positioned: one at the top of the key, another near the right wing beyond the three-point line, and two others near the left corner and just inside the left arc. Meanwhile, Atlanta Hawks’ players are spread defensively: one at the top center, another near the right elbow, one at the center of the paint, and two others at the right low post and left low post. As the play progresses, Cleveland’s players move to positions, with a player coming inside the middle of the right arc, one staying at the top near the center, and another player moving towards the left low post. Hawks adjust defensively: one player guards near the middle of the court, a second near the right elbow inside the paint, and another positioned near the center of the paint. In the next moments, Cleveland players maintain their spacing with one now near the right low post and another in the paint near the left elbow. Hawks’ defending players move slightly, positioning themselves to restrict movement inside and near the arc - one closely guarding near the left block and another at the right elbow. At 6 minutes and 39.6 seconds, Cleveland players focus their positions deeper into the key and near the right block, while Hawks’ defenders continue to maintain a defense-centric approach in the paint and slightly above the free-throw line. Finally, before the 6 minutes and 38.5 seconds mark in the 3rd quarter, during a transition, Cleveland's Tristan Thompson fouls Paul Millsap from the Atlanta Hawks at a point which leads to a stoppage in play. Both teams again reposition with one Hawks’ player now alongside the right low post and others stretching towards center and left positions near the arc.}"
    response4 = predict(prompt, video_data, temperature)
    print(response4)



if __name__ == '__main__':
    test()
