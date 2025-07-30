from airbot_data_collection.common.utils.mcap_utils import (
    McapFlatbufferWriter,
    FlatbufferSchemas,
)
from airbot_data_collection.airbot.samplers.mcap_sampler import (
    AIRBOTMcapDataSampler,
    AIRBOTMcapDataSamplerConfig,
    TaskInfo,
)
from airbot_data_collection.utils import zip
from mcap.writer import Writer
import os
import time
import json
from pydantic import BaseModel
from pydantic_settings import CliApp


class Config(BaseModel):
    """Configuration for the DISCOVERSE to MCAP conversion.
    Args:
        root (str): Root directory containing the task data.
        task_name (str): Name of the task to process.
        output_dir (str): Directory to save the output MCAP files. If not provided,
            it defaults to `<root>/mcap/<task_name>`.
    """

    root: str
    task_name: str
    output_dir: str = ""


config = CliApp.run(Config)

start = time.perf_counter()
directory = f"{config.root}/{config.task_name}"
output_dir = config.output_dir or f"{config.root}/mcap/{config.task_name}"

os.makedirs(output_dir, exist_ok=True)

# find all folders in the directory
folders = [f.path for f in os.scandir(directory) if f.is_dir()]
print(folders)

config = AIRBOTMcapDataSamplerConfig(task_info=TaskInfo(task_name=config.task_name))

for folder in folders:
    fd_base = os.path.basename(folder)
    if not fd_base.isdigit():
        print(f"Skipping folder {folder} as it does not match the expected format.")
        continue
    episode = int(fd_base)
    output_file_path = f"{output_dir}/{episode}.mcap"
    print(f"{output_file_path=}")
    mcap_writer = Writer(output_file_path)
    mcap_writer.start()
    flb_writer = McapFlatbufferWriter()
    flb_writer.set_writter(mcap_writer)
    all_schemas = set(FlatbufferSchemas)
    all_schemas.remove(FlatbufferSchemas.COMPRESSED_IMAGE)
    flb_writer.register_schemas(all_schemas)
    AIRBOTMcapDataSampler.add_config_metadata(mcap_writer, config)
    # find all .mp4 files in the folder
    mp4_files = [
        f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".mp4")
    ]
    print(f"{mp4_files=}")
    # add video attachments
    for mp4_file in mp4_files:
        with open(mp4_file, "rb") as f:
            name = f"/{os.path.basename(mp4_file).removesuffix('.mp4')}/color/image_raw"
            print(f"Adding video attachment: {name}")
            AIRBOTMcapDataSampler.add_video_attachment(
                mcap_writer,
                name,
                f.read(),
            )

    def to_topic(group: str, component: str) -> str:
        return f"/{group}/{component}/joint_state/position"

    # load json dict
    groups = ["lead", "follow"]
    components = ["arm", "eef"]
    slices = [slice(0, 6), slice(6, 7)]
    with open(f"{folder}/obs_action.json") as f:
        act_obs: dict = json.load(f)
        print(f"{act_obs.keys()=}")
        # register joint state channels
        for group in groups:
            for comp in components:
                flb_writer.register_channel(
                    to_topic(group, comp), FlatbufferSchemas.FLOAT_ARRAY
                )
        # add joint states messages
        stamps_ns = []
        for stamp, obs, act in zip(
            act_obs["time"],
            act_obs["obs"]["jq"],
            act_obs["act"],
            strict=True,
        ):
            stamp_ns = int(stamp * 1e9)
            for group, value in zip(groups, [act, obs]):
                for component, slc in zip(components, slices):
                    flb_writer.add_field_array(
                        {"position": to_topic(group, component)},
                        data={"position": value[slc]},
                        publish_time=stamp_ns,
                        log_time=stamp_ns,
                    )
            stamps_ns.append(stamp_ns)
        AIRBOTMcapDataSampler.add_log_stamps_attachment(
            mcap_writer,
            stamps_ns,
        )
    mcap_writer.finish()


print(
    f"Time taken: {time.perf_counter() - start:.3f} seconds of {len(folders)} folders"
)
