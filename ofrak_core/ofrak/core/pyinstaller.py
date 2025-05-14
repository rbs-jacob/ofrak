import io
import logging
import os
import sys
import zlib
import struct
import marshal
from dataclasses import dataclass
from typing import Dict, List

from ofrak.component.unpacker import Unpacker
from ofrak.resource import Resource
from ofrak.core.elf.model import Elf
from ofrak.core.filesystem import File, Folder, FilesystemRoot, SpecialFileType
from ofrak.core.magic import RawMagicPattern

LOGGER = logging.getLogger(__name__)


@dataclass
class PyInstallerResource(Elf, FilesystemRoot):
    """
    An ELF binary that has been packed using PyInstaller.

    It contains both an ELF structure and an embedded filesystem with Python modules.
    """


@dataclass
class PyInstallerTOCEntry:
    """
    Represents a PyInstaller Table of Contents (TOC) entry.
    """

    name: str
    position: int
    cmprsdDataSize: int
    uncmprsdDataSize: int
    cmprsFlag: int
    typeCmprsData: bytes


class PyInstallerUnpacker(Unpacker[None]):
    """
    Unpack a PyInstaller-packed executable.

    This unpacker extracts the embedded files from a PyInstaller archive
    and organizes them into a filesystem structure.
    """

    id = b"PyInstallerUnpacker"
    targets = (PyInstallerResource,)
    children = (File, Folder, SpecialFileType)

    async def unpack(self, resource: Resource, config=None):
        """
        Unpack a PyInstaller executable.

        The process involves:
        1. Locating the PyInstaller CArchive in the ELF binary
        2. Parsing the TOC (Table of Contents)
        3. Extracting files (modules, scripts, etc.)
        4. Creating filesystem resources for each extracted file
        """
        data = await resource.get_data()

        cookie_pos = self._find_cookie_position(data)
        if cookie_pos == -1:
            raise ValueError("Could not find PyInstaller cookie in executable")

        data_stream = io.BytesIO(data)
        data_stream.seek(cookie_pos)

        cookie = data_stream.read(24)
        magic, length, toc_pos, toc_len, pyvers, pymaj, pymin = struct.unpack("!8siiii2s2s", cookie)

        if magic != PYINSTALLER_COOKIE_MAGIC:
            raise ValueError(f"Invalid PyInstaller magic: {magic}")

        pymaj = int(pymaj)
        pymin = int(pymin)

        data_stream.seek(toc_pos)
        toc_data = data_stream.read(toc_len)
        toc_entries = self._parse_toc(toc_data, pymaj, pymin)

        pyc_magic = self._get_pyc_magic(pymaj, pymin)

        extracted_files = {}

        for entry in toc_entries:
            if entry.typeCmprsData == b"d" or entry.typeCmprsData == b"o":
                continue

            data_stream.seek(entry.position)
            entry_data = data_stream.read(entry.cmprsdDataSize)

            if entry.cmprsFlag == 1:
                try:
                    entry_data = zlib.decompress(entry_data)
                except zlib.error:
                    LOGGER.error(f"Failed to decompress {entry.name}")
                    continue

            path = entry.name

            if entry.typeCmprsData == b"s":
                path = f"{entry.name}.pyc"
                entry_data = self._create_pyc_data(entry_data, pyc_magic, pymaj, pymin)

            elif entry.typeCmprsData in (b"M", b"m"):
                path = f"{entry.name}.pyc"
                if not (entry_data[:4] == pyc_magic or entry_data[:2] == b"\r\n"):
                    entry_data = self._create_pyc_data(entry_data, pyc_magic, pymaj, pymin)

            elif entry.typeCmprsData in (b"z", b"Z"):
                pyz_extracted = self._extract_pyz_in_memory(entry_data, pyc_magic, pymaj, pymin)
                for pyz_path, pyz_data in pyz_extracted.items():
                    dir_path = os.path.dirname(pyz_path)
                    if dir_path:
                        await self._ensure_directory(resource, dir_path)
                    await resource.create_child_from_view(
                        File(os.path.basename(pyz_path), None, None), data=pyz_data
                    )

            dirname = os.path.dirname(path)
            if dirname:
                await self._ensure_directory(resource, dirname)

            await resource.create_child_from_view(
                File(os.path.basename(path), None, None), data=entry_data
            )
            extracted_files[path] = entry_data

    async def _ensure_directory(self, resource: Resource, path: str):
        pyinstaller_view = await resource.view_as(PyInstallerResource)
        path_parts = path.split(os.sep)
        current_path = ""

        for part in path_parts:
            if current_path:
                current_path = os.path.join(current_path, part)
            else:
                current_path = part

            if not await pyinstaller_view.get_entry(current_path):
                await pyinstaller_view.add_folder(current_path)

    def _find_cookie_position(self, data: bytes) -> int:
        """
        Find the PyInstaller cookie position in the binary.
        The cookie contains metadata about the embedded archive.
        """
        cookie_pos = data.rfind(PYINSTALLER_COOKIE_MAGIC)
        return cookie_pos

    def _parse_toc(self, toc_data: bytes, pymaj: int, pymin: int) -> List[PyInstallerTOCEntry]:
        """
        Parse the Table of Contents (TOC) data.
        """
        toc_entries = []
        toc_stream = io.BytesIO(toc_data)

        if pymaj >= 3:
            try:
                toc = marshal.load(toc_stream)
            except Exception as e:
                LOGGER.error(f"Failed to unmarshal TOC: {e}")
                return []
        else:
            LOGGER.warning("Python 2.x PyInstaller archives are not supported")
            return []

        for entry in toc:
            name = entry[0]
            position = entry[1]
            length = entry[2]

            if len(entry) >= 7:
                cmprsdDataSize = entry[3]
                uncmprsdDataSize = entry[4]
                cmprsFlag = entry[5]
                typeCmprsData = entry[6]
            else:
                cmprsdDataSize = length
                uncmprsdDataSize = length
                cmprsFlag = 0
                typeCmprsData = b"?"

            name_str = name
            if isinstance(name, bytes):
                try:
                    name_str = name.decode("utf-8")
                except UnicodeDecodeError:
                    name_str = str(name)

            toc_entries.append(
                PyInstallerTOCEntry(
                    name=name_str,
                    position=position,
                    cmprsdDataSize=cmprsdDataSize,
                    uncmprsdDataSize=uncmprsdDataSize,
                    cmprsFlag=cmprsFlag,
                    typeCmprsData=typeCmprsData,
                )
            )

        return toc_entries

    def _get_pyc_magic(self, pymaj: int, pymin: int) -> bytes:
        """
        Get the Python magic number for the given Python version.
        """
        magic_dict = {
            (3, 11): b"\x6f\x0d\r\n",
            (3, 10): b"\x6f\x0c\r\n",
            (3, 9): b"\x6f\x0b\r\n",
            (3, 8): b"\x6f\x0a\r\n",
            (3, 7): b"\x42\x0d\r\n",
            (3, 6): b"\x33\x0d\r\n",
            (3, 5): b"\x16\x0d\r\n",
            (3, 4): b"\xee\x0c\r\n",
            (3, 3): b"\xe3\x0c\r\n",
            (3, 2): b"\x2a\x0c\r\n",
            (3, 1): b"\x03\x0c\r\n",
            (3, 0): b"\xdc\x0b\r\n",
        }

        return magic_dict.get((pymaj, pymin), b"\x00\x00\x00\x00")

    def _create_pyc_data(self, data: bytes, pyc_magic: bytes, pymaj: int, pymin: int) -> bytes:
        pyc_data = bytearray()
        pyc_data.extend(pyc_magic)

        if pymaj >= 3 and pymin >= 7:
            pyc_data.extend(b"\0" * 4)
            pyc_data.extend(b"\0" * 8)
        else:
            pyc_data.extend(b"\0" * 4)
            if pymaj >= 3 and pymin >= 3:
                pyc_data.extend(b"\0" * 4)

        pyc_data.extend(data)
        return bytes(pyc_data)

    def _extract_pyz_in_memory(
        self, pyz_data: bytes, pyc_magic: bytes, pymaj: int, pymin: int
    ) -> Dict[str, bytes]:
        extracted = {}
        pyz_stream = io.BytesIO(pyz_data)

        pyz_magic = pyz_stream.read(4)
        if pyz_magic != b"PYZ\0":
            LOGGER.error(f"Invalid PYZ magic: {pyz_magic}")
            return extracted

        pyz_pyc_magic = pyz_stream.read(4)

        if pymaj != sys.version_info.major or pymin != sys.version_info.minor:
            LOGGER.warning(
                f"Python version mismatch: archive requires {pymaj}.{pymin}, "
                f"but running under {sys.version_info.major}.{sys.version_info.minor}"
            )
            return extracted

        (toc_position,) = struct.unpack("!i", pyz_stream.read(4))

        pyz_stream.seek(toc_position)
        try:
            toc = marshal.load(pyz_stream)
        except Exception as e:
            LOGGER.error(f"Failed to unmarshal PYZ TOC: {e}")
            return extracted

        if isinstance(toc, list):
            toc = dict(toc)

        for key, entry in toc.items():
            file_name = key
            if isinstance(file_name, bytes):
                try:
                    file_name = file_name.decode("utf-8")
                except UnicodeDecodeError:
                    file_name = str(key)

            ispkg, pos, length = entry

            safe_name = file_name.replace("..", "__").replace(".", os.path.sep)

            if ispkg == 1:
                output_path = os.path.join(f"{safe_name}", "__init__.pyc")
            else:
                output_path = f"{safe_name}.pyc"

            pyz_stream.seek(pos)
            try:
                module_data = pyz_stream.read(length)
                module_data = zlib.decompress(module_data)
                pyc_data = self._create_pyc_data(module_data, pyc_magic, pymaj, pymin)
                extracted[output_path] = pyc_data
            except Exception as e:
                LOGGER.error(f"Failed to extract module {file_name}: {e}")

        return extracted


PYINSTALLER_COOKIE_MAGIC = b"MEI\014\013\012\013\016"

RawMagicPattern.register(
    PyInstallerResource, lambda data: len(data) >= 64 and PYINSTALLER_COOKIE_MAGIC in data[-64:]
)
