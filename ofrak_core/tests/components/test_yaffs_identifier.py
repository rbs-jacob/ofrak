from pathlib import Path

import pytest

from ofrak.core.yaffs import Yaffs2Filesystem
from ofrak.ofrak_context import OFRAKContext
from .. import components


ASSETS_DIR = Path(components.ASSETS_DIR)


@pytest.fixture(
    params=[
        "yaffs2_2k_64_le.img",
        "yaffs2_4k_128_le.img",
        "yaffs2_2k_64_be.img",
    ]
)
def yaffs2_asset(request):
    return request.param


async def test_yaffs2_identify(ofrak_context: OFRAKContext, yaffs2_asset: str) -> None:
    """
    Valid YAFFS2 images are tagged as Yaffs2Filesystem.
    """
    asset_path = ASSETS_DIR / yaffs2_asset
    resource = await ofrak_context.create_root_resource_from_file(str(asset_path))
    await resource.identify()
    assert resource.has_tag(
        Yaffs2Filesystem
    ), f"Expected {yaffs2_asset} to be identified as Yaffs2Filesystem"


async def test_yaffs2_not_identified_for_small_data(ofrak_context: OFRAKContext) -> None:
    """
    Data too small to be YAFFS2 should not be identified.
    """
    resource = await ofrak_context.create_root_resource("tiny", b"\x03\x00\x00\x00\x01")
    await resource.identify()
    assert not resource.has_tag(Yaffs2Filesystem)


async def test_yaffs2_not_identified_for_partial_magic(ofrak_context: OFRAKContext) -> None:
    """
    Data with valid header magic but no valid spare area should not be identified.
    """
    data = b"\x03\x00\x00\x00\x01\x00\x00\x00\xff\xff" + b"\x00" * 40000
    resource = await ofrak_context.create_root_resource("partial_magic", data)
    await resource.identify()
    assert not resource.has_tag(Yaffs2Filesystem)
