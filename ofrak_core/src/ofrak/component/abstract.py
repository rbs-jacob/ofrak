import inspect
import logging
from abc import ABC, abstractmethod
from subprocess import CalledProcessError
from typing import (
    Iterable,
    List,
    Optional,
    cast,
    Tuple,
)

from ofrak.component.interface import ComponentInterface
from ofrak.model.component_model import (
    ComponentContext,
    CC,
    ComponentRunResult,
    ComponentConfig,
    ComponentExternalTool,
)
from ofrak.model.data_model import DataPatchesResult
from ofrak.model.job_model import (
    JobRunContext,
)
from ofrak.model.resource_model import (
    ResourceContext,
    MutableResourceModel,
)
from ofrak.model.viewable_tag_model import ResourceViewContext
from ofrak.service.dependency_handler import DependencyHandlerFactory

LOGGER = logging.getLogger(__name__)


class AbstractComponent(ComponentInterface[CC], ABC):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._dependency_handler_factory = DependencyHandlerFactory()
        self._default_config = self.get_default_config()

    @classmethod
    def get_id(cls) -> bytes:
        return cls.id if cls.id is not None else cls.__name__.encode()

    # By default, assume component has no external dependencies
    external_dependencies: Tuple[ComponentExternalTool, ...] = ()

    async def run(
        self,
        job_id: bytes,
        resource_id: bytes,
        job_context: JobRunContext,
        resource_context: ResourceContext,
        resource_view_context: ResourceViewContext,
        config: CC,
    ) -> ComponentRunResult:
        """

        :param job_id:
        :param resource_id:
        :param job_context:
        :param resource_context:
        :param resource_view_context:
        :param config:
        :return: The IDs of all resources modified by this component
        """
        raise NotImplementedError()

    @abstractmethod
    async def _run(self, resource, config: CC):
        raise NotImplementedError()

    async def _save_resources(
        self,
        job_id: bytes,
        mutable_resource_models: Iterable[MutableResourceModel],
        resource_context: ResourceContext,
        resource_view_context: ResourceViewContext,
        job_context: Optional[JobRunContext],
        component_context: ComponentContext,
    ):
        raise NotImplementedError()

    @staticmethod
    def _get_default_config_from_method(component_method) -> Optional[CC]:
        run_signature = inspect.signature(component_method)
        config_arg_type = run_signature.parameters["config"]
        default_arg: CC = config_arg_type.default

        if isinstance(default_arg, ComponentConfig):
            try:
                return cast(CC, default_arg)
            except TypeError as e:
                raise TypeError(
                    f"ComponentConfig subclass {type(default_arg)} is not a dataclass! This is "
                    f"required in order to copy the default config to ensure the default is "
                    f"non-mutable."
                ) from e
        elif default_arg is not None and default_arg is not config_arg_type.empty:
            raise TypeError(
                f"Default config {default_arg} must be either an instance of ComponentConfig, "
                f"None, or left empty!"
            )
        else:
            return None

    async def apply_all_patches(
        self, component_context: ComponentContext
    ) -> List[DataPatchesResult]:
        # Build a list of patches, making sure that there is at most one patch that causes a resize
        patches = []
        for resource_id, tracker in component_context.modification_trackers.items():
            patches.extend(tracker.data_patches)
            tracker.data_patches.clear()
        if len(patches) > 0:
            raise NotImplementedError()
        else:
            return []

    def get_version(self) -> int:
        return 1

    def _log_component_has_run_warning(self, resource):
        LOGGER.warning(
            f"{self.get_id().decode()} has already been run on resource {resource.get_id().hex()}"
        )


class ComponentMissingDependencyError(RuntimeError):
    def __init__(
        self,
        component: ComponentInterface,
        dependency: ComponentExternalTool,
    ):
        if dependency.apt_package:
            apt_install_str = f"\n\tapt installation: apt install {dependency.apt_package}"
        else:
            apt_install_str = ""
        if dependency.brew_package:
            brew_install_str = f"\n\tbrew installation: brew install {dependency.brew_package}"
        else:
            brew_install_str = ""

        super().__init__(
            f"Missing {dependency.tool} tool needed for {type(component).__name__}!"
            f"{apt_install_str}"
            f"{brew_install_str}"
            f"\n\tSee {dependency.tool_homepage} for more info and installation help."
            f"\n\tAlternatively, OFRAK can ignore this component (and any others with missing "
            f"dependencies) so that they will never be run: OFRAK(..., exclude_components_missing_dependencies=True)"
        )

        self.component = component
        self.dependency = dependency


class ComponentSubprocessError(RuntimeError):
    def __init__(self, error: CalledProcessError):
        errstring = (
            f"Command '{error.cmd}' returned non-zero exit status {error.returncode}.\n"
            f"Stderr: {error.stderr}.\n"
            f"Stdout: {error.stdout}."
        )
        super().__init__(errstring)
        self.cmd = error.cmd
        self.cmd_retcode = error.returncode
        self.cmd_stdout = error.stdout
        self.cmd_stderr = error.stderr
