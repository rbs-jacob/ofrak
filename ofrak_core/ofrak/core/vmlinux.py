import tempfile
from dataclasses import dataclass

from vmlinux_to_elf.elf_symbolizer import ElfSymbolizer
from vmlinux_to_elf.vmlinuz_decompressor import obtain_raw_kernel_from_file

from ofrak import Unpacker, Resource
from ofrak.core import GenericBinary, Elf, MagicDescriptionIdentifier
from ofrak.model.component_model import CC


@dataclass
class Vmlinux(GenericBinary):
    """
    Linux kernel Image
    """


class VmlinuxUnpacker(Unpacker[None]):
    targets = (Vmlinux,)
    children = (Elf,)

    async def unpack(self, resource: Resource, config: CC = None) -> None:
        with tempfile.NamedTemporaryFile(suffix=".elf", mode="rb") as temp_elf:
            ElfSymbolizer(obtain_raw_kernel_from_file(await resource.get_data()), temp_elf.name)
            await resource.create_child(tags=(Elf,), data=temp_elf.read())


MagicDescriptionIdentifier.register(
    Vmlinux, lambda s: "linux kernel" in s.lower() and "boot executable image" in s.lower()
)
