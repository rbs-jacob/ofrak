import asyncio
import logging
from typing import Union

from ofrak.component.unpacker import UnpackerError
from ofrak.core import Program, ComplexBlock
from ofrak.model.viewable_tag_model import AttributesType
from ofrak_type import NotFoundError
from ofrak_type.architecture import InstructionSetMode
from ofrak.core.addressable import Addressable
from ofrak.core.architecture import ProgramAttributes
from ofrak.core.basic_block import BasicBlock, BasicBlockUnpacker
from ofrak.core.instruction import (
    Instruction,
    InstructionAnalyzer,
    InstructionRegisterUsageAnalyzer,
    RegisterUsage,
)
from ofrak.resource import Resource, ResourceFactory
from ofrak.service.component_locator_i import ComponentLocatorInterface
from ofrak.service.data_service_i import DataServiceInterface
from ofrak.service.disassembler.disassembler_service_i import (
    DisassemblerServiceInterface,
    DisassemblerServiceRequest,
)
from ofrak.service.resource_service_i import (
    ResourceServiceInterface,
    ResourceFilter,
    ResourceAttributeValueFilter,
)

LOGGER = logging.getLogger(__name__)


class CapstoneBasicBlockUnpacker(BasicBlockUnpacker):
    def __init__(
        self,
        resource_factory: ResourceFactory,
        data_service: DataServiceInterface,
        resource_service: ResourceServiceInterface,
        component_locator: ComponentLocatorInterface,
        disassembler_service: DisassemblerServiceInterface,
    ):
        super().__init__(resource_factory, data_service, resource_service, component_locator)
        self._disassembler_service = disassembler_service

    async def unpack(self, resource: Resource, config=None):
        bb_view = await resource.view_as(BasicBlock)
        bb_data = await resource.get_data()

        program_attrs = await resource.analyze(ProgramAttributes)

        disassemble_request = DisassemblerServiceRequest(
            program_attrs.isa,
            program_attrs.sub_isa,
            program_attrs.bit_width,
            program_attrs.endianness,
            program_attrs.processor,
            bb_view.mode,
            bb_data,
            bb_view.virtual_address,
        )

        instruction_children_created = []

        for disassem_result in await self._disassembler_service.disassemble(disassemble_request):
            operands = " ".join(
                await asyncio.gather(
                    *(
                        get_function_from_address(o, resource)
                        for o in disassem_result.operands.split(" ")
                    )
                )
            )

            instruction_view = Instruction(
                disassem_result.address,
                disassem_result.size,
                f"{disassem_result.mnemonic} {operands}",
                disassem_result.mnemonic,
                operands,
                bb_view.mode,
            )

            instruction_children_created.append(
                bb_view.create_child_region(
                    instruction_view, additional_attributes=(program_attrs,)
                )
            )

        await asyncio.gather(*instruction_children_created)


class CapstoneInstructionAnalyzer(InstructionAnalyzer):
    def __init__(
        self,
        resource_factory: ResourceFactory,
        data_service: DataServiceInterface,
        resource_service: ResourceServiceInterface,
        disassembler_service: DisassemblerServiceInterface,
    ):
        super().__init__(resource_factory, data_service, resource_service)
        self._disassembler_service = disassembler_service

    async def analyze(self, resource: Resource, config=None) -> Instruction:
        parent_block = await resource.get_parent_as_view(Addressable)
        mode: InstructionSetMode
        if parent_block.resource.has_tag(BasicBlock):
            bb_attrs = await parent_block.resource.analyze(AttributesType[BasicBlock])
            mode = bb_attrs.mode  # type: ignore
        else:
            mode = InstructionSetMode.NONE
        instruction_data = await resource.get_data()
        program_attrs = await resource.analyze(ProgramAttributes)

        parent_start_vaddr = parent_block.virtual_address
        instr_offset_in_parent = (await resource.get_data_range_within_parent()).start

        disassemble_request = DisassemblerServiceRequest(
            program_attrs.isa,
            program_attrs.sub_isa,
            program_attrs.bit_width,
            program_attrs.endianness,
            program_attrs.processor,
            mode,
            instruction_data,
            parent_start_vaddr + instr_offset_in_parent,
        )

        try:
            disassem_result = next(
                iter(await self._disassembler_service.disassemble(disassemble_request))
            )
        except StopIteration:
            raise UnpackerError(
                f"Could not disassemble any {program_attrs.isa.name}-"
                f"{program_attrs.bit_width.name}-{program_attrs.endianness.name} instructions "
                f"from bytes {instruction_data.hex()}"
            )

        operands = " ".join(
            await get_function_from_address(o, resource)
            for o in disassem_result.operands.split(" ")
        )

        return Instruction(
            disassem_result.address,
            disassem_result.size,
            f"{disassem_result.mnemonic} {operands}",
            disassem_result.mnemonic,
            operands,
            mode,
        )


async def get_function_from_address(address: Union[int, str], resource: Resource) -> str:
    try:
        address = int(address, 16 if address.startswith("0x") else 10)
    except ValueError:
        return address
    # TODO: Should use get_ancestor_as_view -- may not be the only ancestor
    #  if in (for example) an ELF within an ELF as in a vmlinux file
    program = await resource.get_only_ancestor_as_view(
        Program, r_filter=ResourceFilter.with_tags(Program)
    )
    try:
        function = await program.resource.get_only_descendant_as_view(
            ComplexBlock,
            r_filter=ResourceFilter(
                tags=(ComplexBlock,),
                attribute_filters=(
                    ResourceAttributeValueFilter(ComplexBlock.VirtualAddress, address),
                ),
            ),
        )
        return function.name
    except NotFoundError:
        return hex(address)


class CapstoneInstructionRegisterUsageAnalyzer(InstructionRegisterUsageAnalyzer):
    def __init__(
        self,
        resource_factory: ResourceFactory,
        data_service: DataServiceInterface,
        resource_service: ResourceServiceInterface,
        disassembler_service: DisassemblerServiceInterface,
    ):
        super().__init__(resource_factory, data_service, resource_service)
        self._disassembler_service = disassembler_service

    async def analyze(self, resource: Resource, config=None) -> RegisterUsage:
        program_attrs = await resource.analyze(ProgramAttributes)

        instruction = await resource.view_as(Instruction)

        disassemble_request = DisassemblerServiceRequest(
            program_attrs.isa,
            program_attrs.sub_isa,
            program_attrs.bit_width,
            program_attrs.endianness,
            program_attrs.processor,
            instruction.mode,
            await instruction.resource.get_data(),
            instruction.virtual_address,
        )

        disassem_result = await self._disassembler_service.get_register_usage(disassemble_request)

        return RegisterUsage(
            disassem_result.regs_read,
            disassem_result.regs_written,
        )
