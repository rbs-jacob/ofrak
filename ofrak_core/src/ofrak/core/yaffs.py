import struct
from dataclasses import dataclass
from ofrak.component.identifier import Identifier
from ofrak.core.binary import GenericBinary
from ofrak.core.filesystem import FilesystemRoot
from ofrak.resource import Resource
from ofrak_type.range import Range

PAGE_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
MAX_SPARE_SIZE = 512
NUM_CONFIRM_BLOCKS = 16

# Bytes at the start of the spare area, varies by endianness and ECC settings
SPARE_MAGICS = (
    b"\x00\x00\x10\x00",
    b"\x00\x10\x00\x00",
    b"\xff\xff\x00\x00\x10\x00",
    b"\xff\xff\x00\x10\x00\x00",
)

# First object is probably a dir or file with parent_id=1 and unused
# name_checksum=0xFFFF. At least, that's how Binwalk detects it.
# https://github.com/ReFirmLabs/binwalk/blob/a417b4dcf7420f9153779edf416394d0bb01cdea/src/signatures/yaffs.rs
YAFFS_HEADER_MAGICS = (
    b"\x03\x00\x00\x00\x01\x00\x00\x00\xff\xff",  # LE directory
    b"\x00\x00\x00\x03\x00\x00\x00\x01\xff\xff",  # BE directory
    b"\x01\x00\x00\x00\x01\x00\x00\x00\xff\xff",  # LE file
    b"\x00\x00\x00\x01\x00\x00\x00\x01\xff\xff",  # BE file
)


@dataclass
class Yaffs2Filesystem(GenericBinary, FilesystemRoot):
    """
    Filesystem stored in YAFFS (Yet Another Flash File System) format.
    """


class Yaffs2Identifier(Identifier):
    """
    Identify YAFFSv2 filesystem images by checking for valid YAFFS2 object header
    magic bytes at offset 0, valid spare area magic at the detected page boundary,
    and a valid subsequent object header at the detected block size.
    """

    targets = (GenericBinary,)

    async def identify(self, resource: Resource, config=None) -> None:
        header = await resource.get_data(range=Range(0, 10))
        if len(header) < 10 or header not in YAFFS_HEADER_MAGICS:
            return
        endian = ">" if header[0] == 0x00 else "<"

        # Upper bound: (max_page + max_spare) * max_confirm_blocks + 10
        read_size = (max(PAGE_SIZES) + MAX_SPARE_SIZE) * NUM_CONFIRM_BLOCKS + 10
        data = await resource.get_data(range=Range(0, read_size))

        page_size = detect_page_size(data)
        if page_size == 0:
            return
        spare_size = detect_spare_size(data, page_size, endian)
        if spare_size == 0:
            return

        resource.add_tag(Yaffs2Filesystem)


def detect_page_size(data: bytes) -> int:
    """
    Detect page size by looking for spare magic at known page size offsets.
    """
    for page_size in PAGE_SIZES:
        for magic in SPARE_MAGICS:
            end = page_size + len(magic)
            if end <= len(data) and data[page_size:end] == magic:
                return page_size
    return 0


def detect_spare_size(data: bytes, page_size: int, endian: str) -> int:
    """
    Detect spare size by scanning for the next valid object header after the first page.

    Searches for a valid header at each 4-byte-aligned offset in the spare region
    after the first page.  When a candidate is found, it is validated by checking
    that another valid header exists at a later block_size-aligned offset (file
    objects may have data chunks between headers).
    """
    scan_start = page_size + 4  # skip past spare magic bytes
    scan_end = min(page_size + MAX_SPARE_SIZE, len(data) - 10)
    for offset in range(scan_start, scan_end, 4):
        if parse_obj_header(data[offset:], endian):
            block_size = offset
            # Scan subsequent blocks for another valid header
            for n in range(2, NUM_CONFIRM_BLOCKS):
                later = block_size * n
                if later + 10 > len(data):
                    break
                if parse_obj_header(data[later:], endian):
                    return block_size - page_size
    return 0


def parse_obj_header(data: bytes, endian: str) -> bool:
    if len(data) < 10:
        return False
    obj_type, parent_id, name_checksum = struct.unpack_from(f"{endian}IIH", data, 0)
    return (0 <= obj_type < 6) and parent_id > 0 and name_checksum == 0xFFFF
