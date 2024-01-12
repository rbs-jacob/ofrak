from ofrak_patch_maker.toolchain.gnu_x64 import GNU_X86_64_LINUX_EABI_10_3_0_Toolchain


class GNU_X86_32_LINUX_EABI_10_3_0_Toolchain(GNU_X86_64_LINUX_EABI_10_3_0_Toolchain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._compiler_flags.append("-m32")
        self._assembler_flags.append("--32")
        self._linker_flags.extend(["-m", "elf_i386"])

    @property
    def name(self) -> str:
        return "GNU_X86_32_LINUX_EABI_10_3_0_Toolchain"
