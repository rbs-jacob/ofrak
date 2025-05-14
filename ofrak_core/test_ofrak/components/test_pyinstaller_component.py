import os
import pytest
import logging
from dataclasses import dataclass
from typing import Dict, Set

import test_ofrak.components
from ofrak import OFRAKContext, Resource
from ofrak.core.filesystem import File
from ofrak.core.pyinstaller import PyInstallerResource

from pytest_ofrak.patterns.unpack_verify import UnpackAndVerifyPattern, UnpackAndVerifyTestCase


LOGGER = logging.getLogger(__name__)
ASSETS_DIR = test_ofrak.components.ASSETS_DIR
PYINSTALLER_BINARY = os.path.join(ASSETS_DIR, "pyinstaller_simple_hello")
REGULAR_ELF_BINARY = os.path.join(ASSETS_DIR, "elf", "hello_elf_exec")


@dataclass
class PyInstallerUnpackTestCase(UnpackAndVerifyTestCase[str, bytes]):
    binary_path: str
    min_file_count: int = 0
    expected_patterns: Set[str] = None


PYINSTALLER_UNPACK_TEST_CASES = [
    PyInstallerUnpackTestCase(
        "PyInstaller binary positive test",
        {},
        set(),
        PYINSTALLER_BINARY,
        10,
        {".pyc", "_extracted"},
    ),
    PyInstallerUnpackTestCase(
        "Regular ELF negative test",
        {},
        set(),
        REGULAR_ELF_BINARY,
        0,
        None,
    ),
]


@pytest.mark.skipif_missing_deps([PyInstallerResource])
class TestPyInstaller(UnpackAndVerifyPattern):
    @pytest.fixture(params=PYINSTALLER_UNPACK_TEST_CASES, ids=lambda tc: tc.label)
    async def unpack_verify_test_case(self, request) -> PyInstallerUnpackTestCase:
        return request.param

    @pytest.fixture
    async def root_resource(
        self,
        unpack_verify_test_case: PyInstallerUnpackTestCase,
        ofrak_context: OFRAKContext,
        test_id: str,
    ) -> Resource:
        with open(unpack_verify_test_case.binary_path, "rb") as f:
            binary_data = f.read()

        return await ofrak_context.create_root_resource(test_id, binary_data)

    async def unpack(self, root_resource: Resource):
        try:
            await root_resource.unpack_recursively()
        except Exception as e:
            # Unpacking may fail for non-PyInstaller binaries - that's expected
            if self.unpack_verify_test_case.min_file_count > 0:
                pytest.fail(f"Unpacking failed unexpectedly: {e}")

    async def get_descendants_to_verify(self, unpacked_root_resource: Resource) -> Dict[str, bytes]:
        test_case = self.unpack_verify_test_case

        # Check if the resource was identified as PyInstaller
        is_pyinstaller = unpacked_root_resource.has_tag(PyInstallerResource)

        files = await unpacked_root_resource.get_descendants_as_view(File)

        result = {}
        for file in files:
            path = await file.get_path()
            data = await file.resource.get_data()
            result[path] = data

        # Verify file count
        if test_case.min_file_count > 0:
            if not is_pyinstaller:
                pytest.fail("Resource was expected to be identified as PyInstaller but wasn't")
            assert (
                len(files) >= test_case.min_file_count
            ), f"Expected at least {test_case.min_file_count} files but found {len(files)}"
        elif test_case.min_file_count == 0 and len(files) > 0:
            # If we expect no files but find some, it should only be because it was identified as PyInstaller
            assert is_pyinstaller, "Resource was unpacked but not identified as PyInstaller"

        # Verify file patterns
        if test_case.expected_patterns and len(files) > 0:
            for pattern in test_case.expected_patterns:
                assert any(
                    pattern in path for path in result.keys()
                ), f"No files matching pattern '{pattern}' found"

        return result

    async def verify_descendant(self, unpacked_descendant: bytes, specified_result: bytes):
        pass
