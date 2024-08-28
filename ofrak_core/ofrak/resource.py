import asyncio
import hashlib
import itertools
import logging
import uuid
from typing import (
    BinaryIO,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
    Union,
    Awaitable,
    Callable,
    Pattern,
    overload,
    Dict,
)

from ofrak import Identifier, Unpacker, Analyzer, Modifier, Packer
from ofrak.component.interface import ComponentInterface
from ofrak.model.component_model import CC, ComponentConfig
from ofrak.model.resource_model import (
    ResourceAttributes,
)
from ofrak.model.tag_model import ResourceTag
from ofrak.model.viewable_tag_model import (
    ViewableResourceTag,
    ResourceViewInterface,
)
from ofrak.service.resource_service_i import (
    ResourceFilter,
    ResourceSort,
    ResourceFilterCondition,
    ResourceAttributeRangeFilter,
    ResourceAttributeValueFilter,
    ResourceAttributeValuesFilter,
)
from ofrak_type import NotFoundError
from ofrak_type.range import Range

LOGGER = logging.getLogger(__name__)
RT = TypeVar("RT", bound="ResourceTag")
RA = TypeVar("RA", bound="ResourceAttributes")
RV = TypeVar("RV", bound="ResourceViewInterface")


class Resource:
    """
    Defines methods for interacting with the data and attributes of Resources, the main building
    block of OFRAK.
    """

    components: List[ComponentInterface] = list()

    def __init__(
        self,
        id: Optional[bytes] = None,
        data: Optional[bytes] = None,
        parent: Optional["Resource"] = None,
        tags: Optional[Iterable[ResourceTag]] = None,
        attributes: Optional[Dict[Type[ResourceAttributes], ResourceAttributes]] = None,
    ):
        self.id = id or uuid.uuid4().bytes
        self.parent = parent
        self.tags = set(tags) if tags else set()
        self.attributes = attributes or dict()
        self.data = data or b""
        self.children: List["Resource"] = []

        self._is_modified = False

    def get_id(self) -> bytes:
        """
        :return: This resource's ID
        """
        return self.id

    @property
    def caption(self) -> str:
        captions = []
        for tag in self.get_most_specific_tags():
            captions.append(tag.caption(self.attributes))
        return ", ".join(captions)

    def get_caption(self) -> str:
        return self.caption

    def is_modified(self) -> bool:
        """
        Check if the resource has been modified in this context and is considered "dirty".
        :return: `True` if the resource is modified, `False` otherwise
        """
        return self._is_modified

    async def get_data(self, range: Optional[Range] = None) -> bytes:
        """
        A resource often represents a chunk of underlying binary data. This method returns the
        entire chunk by default; this can be reduced by an optional parameter.

        :param range: A range within the resource's data, relative to the resource's data itself
        (e.g. Range(0, 10) returns the first 10 bytes of the chunk)

        :return: The full range or a partial range of this resource's bytes
        """
        if range:
            return self.data[range.start : range.end]
        return self.data

    async def get_data_length(self) -> int:
        """
        :return: The length of the underlying binary data this resource represents
        """
        return len(self.data)

    async def get_data_range_within_parent(self) -> Range:
        """
        If this resource is "mapped," i.e. its underlying data is defined as a range of its parent's
        underlying data, this method returns the range within the parent resource's data where this
        resource lies. If this resource is not mapped (it is root), it returns a range starting at 0
        with length 0.

        :return: The range of the parent's data which this resource represents
        """
        # raise NotImplementedError()
        return Range(0, 0)

    async def get_data_range_within_root(self) -> Range:
        """
        Does the same thing as `get_data_range_within_parent`, except the range is relative to the
        root.

        :return: The range of the root node's data which this resource represents
        """
        # raise NotImplementedError()
        return Range(0, 0)

    @overload
    async def search_data(
        self,
        query: Pattern[bytes],
        start: Optional[int] = None,
        end: Optional[int] = None,
        max_matches: Optional[int] = None,
    ) -> Tuple[Tuple[int, bytes], ...]:
        ...

    @overload
    async def search_data(
        self,
        query: bytes,
        start: Optional[int] = None,
        end: Optional[int] = None,
        max_matches: Optional[int] = None,
    ) -> Tuple[int, ...]:
        ...

    async def search_data(self, query, start=None, end=None, max_matches=None):
        """
        Search for some data in this resource. The query may be a regex pattern (a return value
        of `re.compile`). If the query is a regex pattern, returns a tuple of pairs with both the
        offset of the match and the contents of the match itself. If the query is plain bytes, a
        list of only the match offsets are returned.

        :param query: Plain bytes to exactly match or a regex pattern to search for
        :param start: Start offset in the data model to begin searching
        :param end: End offset in the data model to stop searching

        :return: A tuple of offsets matching a plain bytes query, or a list of (offset, match) pairs
        for a regex pattern query
        """
        raise NotImplementedError()

    async def save(self):
        """
        If this resource has been modified, update the model stored in the resource service with
        the local changes.

        :raises NotFoundError: If the resource service does not have a model for this resource's ID
        """

    async def run(
        self,
        component_type: Union[Type[ComponentInterface[CC]], ComponentInterface[CC]],
        config: ComponentConfig = None,
    ):
        """
        Run a single component. Runs even if the component has already been run on this resource.

        :param component_type: The component type (may be an interface) to get and run
        :param config: Optional config to pass to the component

        :return: A ComponentRunResult containing information on resources affected by the component
        """
        if isinstance(component_type, type):
            c = component_type(None, None, None)
        else:
            c = component_type
        if isinstance(c, Identifier):
            return await c.identify(self, config)
        elif isinstance(c, Unpacker):
            return await c.unpack(self, config)
        elif isinstance(c, Analyzer):
            result = await c.analyze(self, config)
            self.add_attributes(result)
            for attrs in c.get_attributes_from_results(result):
                self.add_attributes(attrs)
            return result
        elif isinstance(c, Modifier):
            return await c.modify(self, config)
        elif isinstance(c, Packer):
            return await c.pack(self, config)

    async def _auto_run_components(self, component_type):
        return await asyncio.gather(
            *(
                self.run(component, None)
                for component in self.components
                if isinstance(component, component_type) and self.tags & set(component.targets)
            )
        )

    async def unpack(self):
        """
        Unpack the resource.

        :return: A ComponentRunResult containing information on resources affected by the component
        """
        await self._auto_run_components(Identifier)
        await self._auto_run_components(Unpacker)

    async def analyze(self, resource_attributes: Optional[Type[RA]] = None):
        """
        Analyze the resource for a specific resource attribute.

        :param Type[RA] resource_attributes:

        :return:
        """
        if resource_attributes in self.attributes:
            return self.attributes[resource_attributes]
        analyzers = [
            component
            for component in self.components
            if isinstance(component, Analyzer)
            and self.tags & set(component.targets)
            and resource_attributes is not None
            and resource_attributes in component.outputs
        ]
        if resource_attributes and analyzers:
            await asyncio.gather(*(self.run(component, None) for component in analyzers))
            return self.attributes[resource_attributes]
        elif not resource_attributes:
            await self._auto_run_components(Analyzer)

    async def identify(self):
        """
        Run all registered identifiers on the resource, tagging it with matching resource tags.
        """
        await self._auto_run_components(Identifier)

    async def pack(self):
        """
        Pack the resource.

        :return: A ComponentRunResult containing information on resources affected by the component
        """
        await self._auto_run_components(Packer)
        self.children = []

    async def unpack_recursively(
        self,
        blacklisted_components: Iterable[Type[ComponentInterface]] = tuple(),
        do_not_unpack: Iterable[ResourceTag] = tuple(),
    ):
        """
        Automatically unpack this resource and recursively unpack all of its descendants. First
        this resource is unpacked; then, any resource which "valid" tags were added to will also be
        unpacked. New resources created with tags count as resources with new tags. A "valid" tag
        is a tag which is not explicitly ignored via the ``do_not_unpack`` argument.
        The unpacking will only stop when no new "valid" tags have been added in the previous
        iteration. This can lead to a very long unpacking process if it is totally unconstrained.

        :param blacklisted_components: Components which are blocked from running during the
        recursive unpacking, on this resource or any descendants.
        :param do_not_unpack: Do not unpack resources with this tag, and ignore these tags when
        checking if any new tags have been added in this iteration.

        :return: A ComponentRunResult containing information on resources affected by the component
        """
        await self.unpack()
        await asyncio.gather(*(child.unpack_recursively() for child in self.children))

    async def analyze_recursively(self):
        await self.analyze()
        await asyncio.gather(*(child.analyze_recursively() for child in self.children))

    async def pack_recursively(self):
        """
        Recursively pack the resource, starting with its descendants.
        """
        await asyncio.gather(*(child.pack_recursively() for child in self.children))
        await self.pack()

    async def write_to(self, destination: BinaryIO, pack: bool = True):
        """
        Recursively repack resource and write data out to an arbitrary ``BinaryIO`` destination.
        :param destination: Destination for packed resource data
        :return:
        """
        if pack is True:
            await self.pack_recursively()

        destination.write(await self.get_data())

    async def create_child(
        self,
        tags: Iterable[ResourceTag] = None,
        attributes: Iterable[ResourceAttributes] = None,
        data: Optional[bytes] = None,
        data_range: Optional[Range] = None,
    ) -> "Resource":
        """
        Create a new resource as a child of this resource. This method entirely defines the
        child's tags and attributes. This method also defines the child's data semantics:

        A child resource can either be defined in one of three ways:
        1) The resource contains no data ("Dataless" resource). Not used in practice.
        2) As mapping a range of its parent's data ("Mapped" resource). For example, an instruction
        maps a portion of its parent basic block.
        3) Defining its own new, independent data ("Unmapped" resource). For example,
        a file extracted from a zip archive is a child of the zip archive resource, but its data
        does not map to some specific range of that parent archive.

        By default a resource will be defined the third way (unmapped). To specify that the
        resource is a mapped resource, include the optional ``data_range`` parameter set to the
        range of the parent's data which the child maps. That is, `data_range=Range(0,
        10)` creates a resource which maps the first 10 bytes of the parent.
        The optional ``data`` param defines whether to populate the new child's data. It can be used
        only if the data is unmapped. If the child is unmapped, the value of ``data``
        still becomes that child's data, but the parent's data is unaffected. If ``data`` and
        ``data_range`` are both `None` (default), the new child is a dataless resource.

        The following table sums up the possible interactions between ``data`` and ``data_range``:

        |                          | ``data_range`` param not `None`                        | ``data_range`` param `None`                  |
        |--------------------------|--------------------------------------------------------|----------------------------------------------|
        | ``data`` param not `None` | Not allowed                                            | Child unmapped, child's data set to ``data`` |
        | ``data`` param   `None`   | Child mapped, parent's data untouched                  | Child is dataless                            |

        :param tags: [tags][ofrak.model.tag_model.ResourceTag] to add to the new child
        :param attributes: [attributes][ofrak.model.resource_model.ResourceAttributes] to add to
        the new child
        :param data: The binary data for the new child. If `None` and ``data_range`` is `None`,
        the resource has no data. Defaults to `None`.
        :param data_range: The range of the parent's data which the new child maps. If `None` (
        default), the child will not map the parent's data.
        :return:
        """
        if data is not None and data_range is not None:
            raise ValueError(
                "Cannot create a child from both data and data_range. These parameters are "
                "mutually exclusive."
            )
        if data_range is not None:
            data = self.data[data_range.start : data_range.end]
        result = Resource(
            data=data,
            parent=self,
            tags=tags,
            attributes={type(attr): attr for attr in (attributes or ())},
        )
        self.children.append(result)
        return result

    async def create_child_from_view(
        self,
        view: RV,
        data_range: Optional[Range] = None,
        data: Optional[bytes] = None,
        additional_tags: Iterable[ResourceTag] = (),
        additional_attributes: Iterable[ResourceAttributes] = (),
    ) -> "Resource":
        """
        Create a new resource as a child of this resource. The new resource will have tags and
        attributes as defined by the [view][ofrak.model.viewable_tag_model.ViewableResourceTag];
        in this way a view can act as a template to create a new resource.

        The ``additional_tags`` and ``additional_attributes`` can also be used to add more tags
        and attributes beyond what the view contains.

        This method's ``data`` and ``data_range`` parameters have the same semantics as in
        `create_child`, in short:

        |                          | ``data_range`` param not `None`                        | ``data_range`` param `None`                  |
        |--------------------------|--------------------------------------------------------|----------------------------------------------|
        | ``data`` param not `None` | Child mapped, ``data`` patched into child (and parent) | Child unmapped, child's data set to ``data`` |
        | ``data`` param   `None`   | Child mapped, parent's data untouched                  | Child is dataless                            |

        See `create_child` documentation for details.

        :param view: A [resource view][ofrak.resource_view] to pull
        [tags][ofrak.model.tag_model.ResourceTag] and
        [attributes][ofrak.model.resource_model.ResourceAttributes] from to populate the new child
        :param data_range: The range of the parent's data which the new child maps. If `None` (
        default), the child will not map the parent's data.
        :param data: The binary data for the new child. If `None` and ``data_range`` is `None`,
        the resource has no data. Defaults to `None`.
        :param additional_tags: Any [tags][ofrak.model.tag_model.ResourceTag] for the child in
        addition to those from the ``view``
        :param additional_attributes: Any
        [attributes][ofrak.model.resource_model.ResourceAttributes] for the child in addition to
        those from the ``view``
        :return:
        """
        viewable_tag: ViewableResourceTag = type(view)
        new_resource = await self.create_child(
            tags=(viewable_tag, *additional_tags),
            attributes=(*view.get_attributes_instances().values(), *additional_attributes),
            data_range=data_range,
            data=data,
        )
        return new_resource

    async def view_as(self, viewable_tag: Type[RV]) -> RV:
        """
        Provides a specific type of view instance for this resource. The returned instance is an
        object which has some of the information from this same resource, however in a simpler
        interface. This resource instance will itself remain available through the view's
        ``.resource`` property.
        :param viewable_tag: A ViewableResourceTag, which this resource's model must already contain

        :raises ValueError: If the model does not contain this tag, or this tag is not a
        ViewableResourceTag

        :return:
        """
        if viewable_tag not in self.attributes:
            await self.analyze(viewable_tag)
        result = viewable_tag.create(self)
        result.resource = self
        return result

    def add_view(self, view: ResourceViewInterface):
        """
            Add all the attributes composed in a view to this resource, and tag this resource with
        the view type. Calling this is the equivalent of making N ``add_attributes`` calls and
            one ``add_tag`` call (where N is the number of attributes the view is composed of).

            :param view: An instance of a view
        """
        for attributes in view.get_attributes_instances().values():  # type: ignore
            self.add_attributes(attributes)
        self.add_tag(type(view))

    def add_tag(self, *tags: ResourceTag):
        """
        Associate multiple tags with the resource. If the resource already have one of the provided
        tag, the tag is not added. All parent classes of the provided tag that are tags themselves
        are also added.
        """
        for tag in tags:
            # The last three are Resource, ResourceInterface, and object
            for cls in type.mro(tag)[:-3]:
                assert isinstance(cls, ResourceTag)
                self.tags.add(cls)

    def get_tags(self, inherit: bool = True) -> Iterable[ResourceTag]:
        """
        Get a set of tags associated with the resource.
        """
        return self.tags

    def has_tag(self, tag: ResourceTag, inherit: bool = True) -> bool:
        """
        Determine if the resource is associated with the provided tag.
        """
        return tag in self.tags

    def remove_tag(self, tag: ResourceTag):
        self.tags.remove(tag)

    def get_most_specific_tags(self) -> Iterable[ResourceTag]:
        """
        Get all tags associated with the resource from which no other tags on that resource
        inherit. In other words, get the resource's tags that aren't subclassed by other tags on
        the resource.

        For example, for a resource tagged as `Elf`, the result would be just `[Elf]` instead of
        `[Elf, Program, GenericBinary]` that `Resource.get_tags` returns. This is because `Elf`
        inherits from `Program`, which inherits from `GenericBinary`. Even though the resource
        has all of those tags, the most derived class with no other derivatives is the "most
        specific."
        """
        most_specific_tags = set(self.tags)
        for tag in [
            base for tag in self.tags for base in cast(Iterable[ResourceTag], tag.__bases__)
        ]:
            most_specific_tags.discard(tag)
        return most_specific_tags

    def add_attributes(self, *attributes: ResourceAttributes):
        """
        Add the provided attributes to the resource. If the resource already have the
        provided attributes classes, they are replaced with the provided one.
        """
        for attrs in attributes:
            self.attributes[type(attrs)] = attrs

    def has_attributes(self, attributes_type: Type[ResourceAttributes]) -> bool:
        """
        Check if this resource has a value for the given attributes type.
        :param attributes_type:
        :return:
        """
        return attributes_type in self.attributes

    def get_attributes(self, attributes_type: Type[RA]) -> RA:
        """
        If this resource has attributes matching the given type, return the value of those
        attributes. Otherwise returns `None`.
        :param attributes_type:
        :return:
        """
        if attributes_type not in self.attributes:
            raise NotFoundError()
        return self.attributes[attributes_type]

    def remove_attributes(self, attributes_type: Type[ResourceAttributes]):
        """
        Remove the value of a given attributes type from this resource, if there is such a value.
        If the resource does not have a value for the given attributes type, do nothing.
        :param attributes_type:
        :return:
        """
        del self.attributes[attributes_type]

    def queue_patch(
        self,
        patch_range: Range,
        data: bytes,
    ):
        """
        Replace the data within the provided range with the provided data. This operation may
        shrink, expand or leave untouched the resource's data. Patches are queued up to be
        applied, and will only be applied to the resource's data after the component this was
        called from exits.

        :param patch_range: The range of binary data in this resource to replace
        :param data: The bytes to replace part of this resource's data with
        :return:
        """
        self.data = self.data[: patch_range.start] + data + self.data[patch_range.end :]

    async def get_parent_as_view(self, v_type: Type[RV]) -> RV:
        """
        Get the parent of this resource. The parent will be returned as an instance of the given
        [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param v_type: The type of [view][ofrak.resource] to get the parent as
        """
        parent_r = await self.get_parent()
        return await parent_r.view_as(v_type)

    async def get_parent(self) -> "Resource":
        """
        Get the parent of this resource.
        """
        if self.parent is None:
            raise RuntimeError("Parent is None")
        return self.parent

    async def get_ancestors(
        self,
        r_filter: ResourceFilter = None,
    ) -> Iterable["Resource"]:
        """
        Get all the ancestors of this resource. May optionally filter the ancestors so only those
        matching certain parameters are returned.

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter was provided and no resources match the provided filter
        """
        result = []
        parent = self.parent
        if r_filter and r_filter.include_self:
            result.append(self)
        while parent is not None:
            if run_attribute_filter(r_filter, parent):
                result.append(parent)
            parent = parent.parent
        return result

    async def get_only_ancestor_as_view(
        self,
        v_type: Type[RV],
        r_filter: ResourceFilter,
    ) -> RV:
        """
        Get the only ancestor of this resource which matches the given filter. The ancestor will be
        returned as an instance of the given
        [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If more or fewer than one ancestor matches ``r_filter``
        """
        ancestor_r = await self.get_only_ancestor(r_filter)
        return await ancestor_r.view_as(v_type)

    async def get_only_ancestor(self, r_filter: ResourceFilter) -> "Resource":
        """
        Get the only ancestor of this resource which matches the given filter.

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:
        """
        return list(await self.get_ancestors(r_filter=r_filter))[0]

    async def get_descendants_as_view(
        self,
        v_type: Type[RV],
        max_depth: int = -1,
        r_filter: ResourceFilter = None,
        r_sort: ResourceSort = None,
    ) -> Iterable[RV]:
        """
        Get all the descendants of this resource. May optionally filter the descendants so only
        those matching certain parameters are returned. May optionally sort the descendants by
        an indexable attribute value key. The descendants will be returned as an
        instance of the given [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param v_type: The type of [view][ofrak.resource] to get the descendants as
        :param max_depth: Maximum depth from this resource to search for descendants; if -1,
        no maximum depth
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :param r_sort: Specifies which indexable attribute to use as the key to sort and the
        direction to sort
        :return:

        :raises NotFoundError: If a filter was provided and no resources match the provided filter
        """
        descendants = await self.get_descendants(max_depth, r_filter, r_sort)
        return await asyncio.gather(*(r.view_as(v_type) for r in descendants))

    async def get_descendants(
        self,
        max_depth: int = -1,
        r_filter: ResourceFilter = None,
        r_sort: ResourceSort = None,
    ) -> Iterable["Resource"]:
        """
        Get all the descendants of this resource. May optionally filter the descendants so only
        those matching certain parameters are returned. May optionally sort the descendants by
        an indexable attribute value key.

        :param max_depth: Maximum depth from this resource to search for descendants; if -1,
        no maximum depth
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :param r_sort: Specifies which indexable attribute to use as the key to sort and the
        direction to sort
        :return:

        :raises NotFoundError: If a filter was provided and no resources match the provided filter
        """
        # TODO: Handle sorting
        if max_depth == 0:
            return []
        result = []
        if r_filter and r_filter.include_self:
            result.append(self)
            r_filter.include_self = False
        for child in self.children:
            if run_attribute_filter(r_filter, child):
                result.append(child)
            result.extend(
                await child.get_descendants(
                    max_depth=max_depth - 1, r_filter=r_filter, r_sort=r_sort
                )
            )
        return itertools.islice(result, None)

    async def get_only_descendant_as_view(
        self,
        v_type: Type[RV],
        max_depth: int = -1,
        r_filter: ResourceFilter = None,
    ) -> RV:
        """
        If a filter is provided, get the only descendant of this resource which matches the given
        filter. If a filter is not provided, gets the only descendant of this resource. The
        descendant will be returned as an instance of the given
        [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param v_type: The type of [view][ofrak.resource] to get the descendant as
        :param max_depth: Maximum depth from this resource to search for descendants; if -1,
        no maximum depth
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one descendant matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple descendant
        """
        descendant_r = await self.get_only_descendant(max_depth, r_filter)
        return await descendant_r.view_as(v_type)

    async def get_only_descendant(
        self,
        max_depth: int = -1,
        r_filter: ResourceFilter = None,
    ) -> "Resource":
        """
        If a filter is provided, get the only descendant of this resource which matches the given
        filter. If a filter is not provided, gets the only descendant of this resource.

        :param max_depth: Maximum depth from this resource to search for descendants; if -1,
        no maximum depth
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one descendant matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple descendant
        """
        return list(await self.get_descendants(max_depth=max_depth, r_filter=r_filter))[0]

    async def get_only_sibling_as_view(
        self,
        v_type: Type[RV],
        r_filter: ResourceFilter = None,
    ) -> RV:
        """
        If a filter is provided, get the only sibling of this resource which matches the given
        filter. If a filter is not provided, gets the only sibling of this resource. The sibling
        will be returned as an instance of the given
        [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].
        :param v_type: The type of [view][ofrak.resource] to get the sibling as
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one sibling matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple siblings
        """
        sibling_r = await self.get_only_sibling(r_filter)
        return await sibling_r.view_as(v_type)

    async def get_only_sibling(self, r_filter: ResourceFilter = None) -> "Resource":
        """
        If a filter is provided, get the only sibling of this resource which matches the given
        filter. If a filter is not provided, gets the only sibling of this resource.

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one sibling matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple siblings
        """
        return next(
            sibling
            for sibling in await (await self.get_parent()).get_children(r_filter=r_filter)
            if sibling != self
        )

    async def get_children(
        self,
        r_filter: ResourceFilter = None,
        r_sort: ResourceSort = None,
    ) -> Iterable["Resource"]:
        """
        Get all the children of this resource. May optionally sort the children by an
        indexable attribute value key. May optionally filter the children so only those
        matching certain parameters are returned.

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :param r_sort: Specifies which indexable attribute to use as the key to sort and the
        direction to sort
        :return:

        :raises NotFoundError: If a filter was provided and no resources match the provided filter
        """
        return await self.get_descendants(1, r_filter, r_sort)

    async def get_children_as_view(
        self,
        v_type: Type[RV],
        r_filter: ResourceFilter = None,
        r_sort: ResourceSort = None,
    ) -> Iterable[RV]:
        """
        Get all the children of this resource. May optionally filter the children so only those
        matching certain parameters are returned. May optionally sort the children by an
        indexable attribute value key. The children will be returned as an instance of
        the given [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param v_type: The type of [view][ofrak.resource] to get the children as
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :param r_sort: Specifies which indexable attribute to use as the key to sort and the
        direction to sort
        :return:

        :raises NotFoundError: If a filter was provided and no resources match the provided filter
        """
        return await self.get_descendants_as_view(v_type, 1, r_filter, r_sort)

    async def get_only_child(self, r_filter: ResourceFilter = None) -> "Resource":
        """
        If a filter is provided, get the only child of this resource which matches the given
        filter. If a filter is not provided, gets the only child of this resource.

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one child matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple children
        """
        return await self.get_only_descendant(1, r_filter)

    async def get_only_child_as_view(self, v_type: Type[RV], r_filter: ResourceFilter = None) -> RV:
        """
        If a filter is provided, get the only child of this resource which matches the given
        filter. If a filter is not provided, gets the only child of this resource. The child will
        be returned as an instance of the given
        [viewable tag][ofrak.model.viewable_tag_model.ViewableResourceTag].

        :param v_type: The type of [view][ofrak.resource] to get the child as
        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :return:

        :raises NotFoundError: If a filter is provided and more or fewer than one child matches
        ``r_filter``
        :raises NotFoundError: If a filter is not provided and this resource has multiple children
        """
        return await self.get_only_descendant_as_view(v_type, 1, r_filter)

    async def delete(self):
        """
        Delete this resource and all of its descendants.

        :return:
        """
        for child_r in await self.get_children():
            await child_r.delete()
        self.parent.remove_child(self)

    def remove_child(self, child):
        if child not in self.children:
            return
        self.children.remove(child)

    async def flush_data_to_disk(self, path: str, pack: bool = True):
        """
        Recursively repack the resource and write its data out to a file on disk. If this is a
        dataless resource, creates an empty file.

        :param path: Path to the file to write out to. The file is created if it does not exist.
        """
        if pack is True:
            await self.pack_recursively()

        data = await self.get_data()
        if data is not None:
            with open(path, "wb") as f:
                f.write(data)
        else:
            # Create empty file
            with open(path, "wb") as f:
                pass

    def __repr__(self):
        properties = [
            f"resource_id={self.id.hex()}",
            f"tag=[{','.join([tag.__name__ for tag in self.tags])}]",
        ]
        return f"{type(self).__name__}(" + ", ".join(properties) + f")"

    async def summarize(self) -> str:
        """
        Create a string summary of this resource, including specific tags, attribute types,
        and the data offsets of this resource in the parent and root (if applicable).

        Not that this is not a complete string representation of the resource: not all tags are
        included, and only the types of attributes are included, not their values. It is a
        summary which gives a high level overview of the resource.
        """
        return await _default_summarize_resource(self)

    async def summarize_tree(
        self,
        r_filter: ResourceFilter = None,
        r_sort: ResourceSort = None,
        indent: str = "",
        summarize_resource_callback: Optional[Callable[["Resource"], Awaitable[str]]] = None,
    ) -> str:
        """
        Create a string summary of this resource and its (optionally filtered and/or sorted)
        descendants. The summaries of each resource are the same as the result of
        [summarize][ofrak.resource.Resource.summarize], organized into a tree structure.
        If a filter parameter is provided, it is applied recursively: the children of this
        resource will be filtered, then only those children matching
        the filter be displayed, and then the same filter will be applied to their children,
        etc. For example,

        :param r_filter: Contains parameters which resources must match to be returned, including
        any tags it must have and/or values of indexable attributes
        :param r_sort: Specifies which indexable attribute to use as the key to sort and the
        direction to sort
        """
        SPACER_BLANK = "   "
        SPACER_LINE = "───"

        if summarize_resource_callback is None:
            summarize_resource_callback = _default_summarize_resource

        children = cast(
            List[Resource], list(await self.get_children(r_filter=r_filter, r_sort=r_sort))
        )

        if children:
            if indent == "":
                tree_string = "┌"
            else:
                tree_string = "┬"
        else:
            tree_string = "─"

        tree_string += f"{await summarize_resource_callback(self)}\n"

        # All children but the last should display as a "fork" in the drop-down tree
        # After the last child, a vertical line should not be drawn as part of the indent
        # Both of those needs are handled here
        child_formatting: List[Tuple[str, str]] = [
            ("├", indent + "│" + SPACER_BLANK) for _ in children[:-1]
        ]
        child_formatting.append(("└", indent + " " + SPACER_BLANK))

        for child, (branch_symbol, child_indent) in zip(children, child_formatting):
            child_tree_string = await child.summarize_tree(
                r_filter=r_filter,
                r_sort=r_sort,
                indent=child_indent,
                summarize_resource_callback=summarize_resource_callback,
            )
            tree_string += f"{indent}{branch_symbol}{SPACER_LINE}{child_tree_string}"
        return tree_string


async def _default_summarize_resource(resource: Resource) -> str:
    attributes_info = ", ".join(attrs_type.__name__ for attrs_type in resource.attributes)

    if resource.data:
        root_data_range = await resource.get_data_range_within_root()
        parent_data_range = await resource.get_data_range_within_parent()
        data = await resource.get_data()
        if len(data) <= 128:
            # Convert bytes to string to check .isprintable without doing .decode. Note that
            # not all ASCII is printable, so we have to check both decodable and printable
            raw_data_str = "".join(map(chr, data))
            if raw_data_str.isascii() and raw_data_str.isprintable():
                data_string = f'data_ascii="{data.decode("ascii")}"'
            else:
                data_string = f"data_hex={data.hex()}"
        else:
            sha256 = hashlib.sha256()
            sha256.update(data)
            data_string = f"data_hash={sha256.hexdigest()[:8]}"
        data_info = (
            f", global_offset=({hex(root_data_range.start)}-{hex(root_data_range.end)})"
            f", parent_offset=({hex(parent_data_range.start)}-{hex(parent_data_range.end)})"
            f", {data_string}"
        )
    else:
        data_info = ""
    return (
        f"{resource.get_id().hex()}: [caption=({resource.get_caption()}), "
        f"attributes=({attributes_info}){data_info}]"
    )


class ResourceFactory:
    def __init__(self, *args, **kwargs):
        pass


def run_attribute_filter(r_filter, resource):
    passed_tag_filter = True
    if r_filter and r_filter.tags:
        if r_filter.tags_condition == ResourceFilterCondition.AND and not all(
            resource.has_tag(t) for t in r_filter.tags
        ):
            passed_tag_filter = False
        elif r_filter.tags_condition == ResourceFilterCondition.OR and not any(
            resource.has_tag(t) for t in r_filter.tags
        ):
            passed_tag_filter = False
    if not passed_tag_filter:
        return False
    passed_attribute_filters = True
    if r_filter and r_filter.attribute_filters:
        for attribute_filter in r_filter.attribute_filters:
            value = attribute_filter.attribute.get_value(resource)
            if isinstance(attribute_filter, ResourceAttributeRangeFilter) and (
                value < attribute_filter.min or value >= attribute_filter.max
            ):
                passed_attribute_filters = False
                break
            elif (
                isinstance(attribute_filter, ResourceAttributeValueFilter)
                and value != attribute_filter.value
            ):
                passed_attribute_filters = False
                break
            elif (
                isinstance(attribute_filter, ResourceAttributeValuesFilter)
                and value not in attribute_filter.values
            ):
                passed_attribute_filters = False
                break
    return passed_tag_filter and passed_attribute_filters
