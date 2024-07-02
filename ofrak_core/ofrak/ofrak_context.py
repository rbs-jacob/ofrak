import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import webbrowser
from base64 import b64decode
from pydoc import pager
from types import ModuleType
from typing import Type, Any, Awaitable, Callable, List, Iterable, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import ofrak_patch_maker

from ofrak_type import InvalidStateError
from synthol.injector import DependencyInjector

from ofrak.component.interface import ComponentInterface
from ofrak.core.binary import GenericBinary
from ofrak.core.filesystem import File, FilesystemRoot
from ofrak.model.component_model import ClientComponentContext
from ofrak.model.resource_model import (
    ResourceModel,
    EphemeralResourceContextFactory,
)
from ofrak.model.tag_model import ResourceTag
from ofrak.model.viewable_tag_model import ResourceViewContext
from ofrak.resource import Resource, ResourceFactory
from ofrak.service.abstract_ofrak_service import AbstractOfrakService
from ofrak.service.component_locator_i import ComponentLocatorInterface
from ofrak.service.data_service_i import DataServiceInterface
from ofrak.service.id_service_i import IDServiceInterface
from ofrak.service.job_service_i import JobServiceInterface
from ofrak.service.resource_service_i import ResourceServiceInterface

LOGGER = logging.getLogger("ofrak")
DEFAULT_OFRAK_LOG_FILE = os.path.join(tempfile.gettempdir(), "ofrak.log")

COMMUNITY_LICENSE_DATA = """{
  "license_type": "Community License",
  "name": "OFRAK Community",
  "email": "ofrak@redballoonsecurity.com",
  "phone_number": null,
  "date": "1719848612",
  "expiration_date": null,
  "signature": "C1m/AuHocQdW1WniFgDZpZuYJoCn0wwgtVhU3BDNWHdBkWuRcy2sJtYZU1AX6GwAnCEW6x2wmMBfMRY1f5wuCg==",
  "agreement": "RED BALLOON SECURITY, INC., A DELAWARE CORPORATION, WITH AN ADDRESS AT 639 11TH AVENUE, 4TH FLOOR, NEW YORK, NY 10036, USA (\\"RED BALLOON\\") IS ONLY WILLING TO LICENSE OFRAK AND RELATED DOCUMENTATION PURSUANT TO THE OFRAK PRO LICENSE AGREEMENT (COLLECTIVELY WITH THIS REGISTRATION FORM, THE \\"AGREEMENT\\"). READ THIS AGREEMENT CAREFULLY BEFORE DOWNLOADING AND INSTALLING AND USING OFRAK.  BY CLICKING ON THE \\"ACCEPT\\" BUTTON ON THIS REGISTRATION FORM \\"REGISTRATION FORM\\"), OR OTHERWISE ACCESSING, INSTALLING, COPYING OR OTHERWISE USING OFRAK, YOU (\\"LICENSEE\\") AGREE THAT THIS REGISTRATION FORM SHALL BE DEEMED TO BE MUTUALLY EXECUTED AND THIS REGISTRATION FORM SHALL BE INCORPORATED INTO AND BECOME A MATERIAL PART OF THE AGREEMENT BETWEEN LICENSEE AND RED BALLOON LOCATED AT www.ofrak.com/license. YOU REPRESENT THAT YOU ARE AUTHORIZED TO ACCEPT THIS AGREEMENT ON BEHALF OF LICENSEE.  IF LICENSEE DOES NOT AGREE TO THE FOREGOING TERMS AND CONDITIONS, DO NOT CLICK ON THE ACCEPT BUTTON, OR OTHERWISE ACCESS, INSTALL, COPY OR USE OFRAK."
}"""
RBS_PUBLIC_KEY = b"r\xcf\xb2\xe7\x17Y\x05*\x0e\xe3+\x00\x16\xd3\xd6\xf7\xa7\xd8\xd7\xfdV\x91\xa7\x88\x93\xe9\x9a\x8a\x05q\xd3\xbd"
LICENSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "license.json"))


class OFRAKContext:
    def __init__(
        self,
        injector: DependencyInjector,
        resource_factory: ResourceFactory,
        component_locator: ComponentLocatorInterface,
        id_service: IDServiceInterface,
        data_service: DataServiceInterface,
        resource_service: ResourceServiceInterface,
        job_service: JobServiceInterface,
        all_ofrak_services: List[AbstractOfrakService],
    ):
        self.injector = injector
        self.resource_factory = resource_factory
        self.component_locator = component_locator
        self.id_service = id_service
        self.data_service = data_service
        self.resource_service = resource_service
        self.job_service = job_service
        self._all_ofrak_services = all_ofrak_services
        self._resource_context_factory = EphemeralResourceContextFactory()

    async def create_root_resource(
        self, name: str, data: bytes, tags: Iterable[ResourceTag] = (GenericBinary,)
    ) -> Resource:
        job_id = self.id_service.generate_id()
        resource_id = self.id_service.generate_id()
        data_id = resource_id

        await self.job_service.create_job(job_id, name)
        await self.data_service.create_root(data_id, data)
        resource_model = await self.resource_service.create(
            ResourceModel.create(resource_id, data_id, tags=tags)
        )
        root_resource = await self.resource_factory.create(
            job_id,
            resource_model.id,
            self._resource_context_factory.create(),
            ResourceViewContext(),
            ClientComponentContext(),
        )
        return root_resource

    async def create_root_resource_from_file(self, file_path: str) -> Resource:
        full_file_path = os.path.abspath(file_path)
        with open(full_file_path, "rb") as f:
            root_resource = await self.create_root_resource(
                os.path.basename(full_file_path), f.read(), (File,)
            )
        root_resource.add_view(
            File(
                os.path.basename(full_file_path),
                os.lstat(full_file_path),
                FilesystemRoot._get_xattr_map(full_file_path),
            )
        )
        await root_resource.save()
        return root_resource

    async def create_root_resource_from_directory(self, dir_path: str) -> Resource:
        full_dir_path = os.path.abspath(dir_path)
        root_resource = await self.create_root_resource(
            os.path.basename(full_dir_path), b"", (FilesystemRoot,)
        )
        root_resource_v = await root_resource.view_as(FilesystemRoot)
        await root_resource_v.initialize_from_disk(full_dir_path)
        return root_resource

    async def start_context(self):
        if "_ofrak_context" in globals():
            raise InvalidStateError(
                "Cannot start OFRAK context as a context has already been started in this process!"
            )
        globals()["_ofrak_context"] = self
        await asyncio.gather(*(service.run() for service in self._all_ofrak_services))

    async def shutdown_context(self):
        if "_ofrak_context" in globals():
            del globals()["_ofrak_context"]
        await asyncio.gather(*(service.shutdown() for service in self._all_ofrak_services))
        logging.shutdown()


class OFRAK:
    DEFAULT_LOG_LEVEL = logging.WARNING

    def __init__(
        self,
        logging_level: int = DEFAULT_LOG_LEVEL,
        exclude_components_missing_dependencies: bool = False,
        license_check: bool = True,
    ):
        """
        Set up the OFRAK environment that a script will use.

        :param logging_level: Logging level of OFRAK instance (logging.DEBUG, logging.WARNING, etc.)
        :param exclude_components_missing_dependencies: When initializing OFRAK, check each component's dependency and do
        not use any components missing some dependencies
        """
        logging.basicConfig(level=logging_level, format="[%(filename)15s:%(lineno)5s] %(message)s")
        logging.getLogger().addHandler(logging.FileHandler(DEFAULT_OFRAK_LOG_FILE))
        logging.getLogger().setLevel(logging_level)
        logging.captureWarnings(True)
        self.injector = DependencyInjector()
        self._discovered_modules: List[ModuleType] = []
        self._exclude_components_missing_dependencies = exclude_components_missing_dependencies
        self._id_service: Optional[IDServiceInterface] = None

        if license_check:
            self._do_license_check()

    def discover(
        self,
        module: ModuleType,
        blacklisted_interfaces: Iterable[Type] = (),
        blacklisted_modules: Iterable[Any] = (),
    ):
        self.injector.discover(module, blacklisted_interfaces, blacklisted_modules)
        self._discovered_modules.append(module)

    def set_id_service(self, service: IDServiceInterface):
        self._id_service = service

    async def create_ofrak_context(self) -> OFRAKContext:
        """
        Create the OFRAKContext and start all its services.
        """
        self._setup()
        component_locator = await self.injector.get_instance(ComponentLocatorInterface)

        resource_factory = await self.injector.get_instance(ResourceFactory)
        components = await self._get_discovered_components()
        component_locator.add_components(components, self._discovered_modules)

        id_service = await self.injector.get_instance(IDServiceInterface)
        data_service = await self.injector.get_instance(DataServiceInterface)
        resource_service = await self.injector.get_instance(ResourceServiceInterface)
        job_service = await self.injector.get_instance(JobServiceInterface)
        all_services = await self.injector.get_instance(List[AbstractOfrakService])

        ofrak_context = OFRAKContext(
            self.injector,
            resource_factory,
            component_locator,
            id_service,
            data_service,
            resource_service,
            job_service,
            all_services,
        )
        await ofrak_context.start_context()
        return ofrak_context

    # TODO: Typehints here do not properly accept functions with variable args
    async def run_async(self, func: Callable[["OFRAKContext", Any], Awaitable[None]], *args):
        ofrak_context = await self.create_ofrak_context()
        start = time.time()
        try:
            await func(ofrak_context, *args)
        finally:
            await ofrak_context.shutdown_context()
            print(f"It took {time.time() - start:.3f} seconds to run the OFRAK script")

    # TODO: Typehints here do not properly accept functions with variable args
    def run(self, func: Callable[["OFRAKContext", Any], Awaitable[None]], *args):
        asyncio.get_event_loop().run_until_complete(self.run_async(func, *args))

    def _setup(self):
        """Discover common OFRAK services and components."""
        import ofrak

        self.discover(ofrak)
        self.discover(ofrak_patch_maker)

        if self._id_service:
            self.injector.bind_instance(self._id_service)

    async def _get_discovered_components(self) -> List[ComponentInterface]:
        all_discovered_components = await self.injector.get_instance(List[ComponentInterface])
        if not self._exclude_components_missing_dependencies:
            return all_discovered_components
        LOGGER.debug(
            "`exclude_components_missing_dependencies` set True; checking each discovered component's dependencies are "
            "installed"
        )
        components_missing_deps = []
        audited_components = []
        for component in all_discovered_components:
            if all(
                await asyncio.gather(
                    *[dep.is_tool_installed() for dep in component.external_dependencies]
                )
            ):
                audited_components.append(component)
            else:
                components_missing_deps.append(component)

        LOGGER.warning(
            f"Skipped registering the following components due to missing dependencies: "
            f"{', '.join(type(c).__name__ for c in components_missing_deps)}. Run `python3 -m "
            f"ofrak deps --missing-only` for more details."
        )

        return audited_components

    @staticmethod
    def _write_license(data):
        _data = json.loads(data)
        pager(
            "Read the license agreement below.\n\n"
            + _data["agreement"]
            + '\n\nPress "q" to continue.'
        )
        agreement = None
        while agreement is None or agreement.lower() != "i agree":
            agreement = input('Type "I agree" to agree to the license terms: ')
        with open(LICENSE_PATH, "w") as f:
            f.write(data)

    @staticmethod
    def _license_selection():
        license_type = choose(
            "How will you use OFRAK?",
            "I will use OFRAK for personal projects",
            "I will use OFRAK at work",
        )
        if license_type == 0:
            OFRAK._write_license(COMMUNITY_LICENSE_DATA)
            return
        find_or_buy = choose(
            "Do you already have an OFRAK license?",
            "Obtain a license from Red Balloon Security",
            "Choose a license file on disk",
            "Paste license data in directly",
        )
        if find_or_buy == 0:
            webbrowser.open("https://ofrak.com/license/")
        elif find_or_buy == 1:
            license_path = input("Path to license file: ")
            with open(license_path) as f:
                OFRAK._write_license(f.read())
        else:
            print(
                "Paste in the contents of the OFRAK license and press ctrl+d when done",
                end="\n\n",
            )
            OFRAK._write_license(sys.stdin.read())

    @staticmethod
    def _do_license_check():
        """
        License check function raises one of several possible exceptions if any
        part of the license is invalid.

        If, for some reason, you're trying to bypass, investigate, or otherwise
        reverse-engineer this license check, you might be a good candidate to
        work at Red Balloon Security – we're hiring! Check out our jobs page
        for more info:

        https://redballoonsecurity.com/company/careers/
        """
        if not os.path.exists(LICENSE_PATH):
            OFRAK._license_selection()

        with open(LICENSE_PATH) as f:
            license_data = json.load(f)

        print(f"\nUsing OFRAK with license type: {license_data['license_type']}\n")

        # Canonicalize license data and serialize to validate signature. Signed
        # fields must be ordered to ensure data is serialized consistently for
        # signature validation.
        signed_fields = ["name", "date", "expiration_date", "email"]  # TODO: Add fields
        to_validate = json.dumps([(k, license_data[k]) for k in signed_fields]).encode("utf-8")

        key = Ed25519PublicKey.from_public_bytes(RBS_PUBLIC_KEY)
        key.verify(b64decode(license_data["signature"]), to_validate)
        if (
            license_data["expiration_date"] is not None
            and int(license_data["expiration_date"]) < time.time()
        ):
            raise RuntimeError("OFRAK license expired! Please purchase a pro license.")


def choose(prompt, *options: str) -> int:
    print(prompt)
    for i, option in enumerate(options):
        print(f"[{i + 1}] {option}")
    selection = 0
    while not (1 <= selection <= len(options)):
        try:
            selection = int(input(f"Enter an option (1-{len(options)}): "))
        except (ValueError, TypeError):
            continue
    return selection - 1


def get_current_ofrak_context() -> OFRAKContext:
    # TODO: This is a brittle MVP, creating multiple simultaneous contexts in a single process
    #  will probably break it!
    ctx = globals().get("_ofrak_context")
    if ctx is None:
        raise InvalidStateError("Not in an OFRAK context!")
    else:
        return ctx
