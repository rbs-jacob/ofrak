import bisect
import dataclasses
import inspect
import logging
import pickle
import sqlite3

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Set, Optional, Iterable, Tuple, Any, TypeVar, Generic

from sortedcontainers import SortedList

from ofrak.model.resource_model import (
    ResourceModel,
    ResourceModelDiff,
    ResourceIndexedAttribute,
    ResourceAttributeDependency,
)
from ofrak.model.tag_model import ResourceTag
from ofrak.service.resource_service_i import (
    ResourceServiceInterface,
    ResourceFilter,
    ResourceSort,
    ResourceAttributeFilter,
    ResourceAttributeRangeFilter,
    ResourceAttributeValueFilter,
    ResourceAttributeValuesFilter,
    ResourceSortDirection,
    ResourceFilterCondition,
    ResourceServiceWalkError,
)
from ofrak_type import AlreadyExistError, NotFoundError
from ofrak_type.range import Range

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", str, int, float, bytes)


class LowValue:
    def __gt__(self, other):
        return False


class HighValue:
    def __lt__(self, other):
        return False


LOW_VALUE = LowValue()
HIGH_VALUE = HighValue()


class ResourceNode:
    model: ResourceModel
    parent: Optional["ResourceNode"]
    _children: Dict["ResourceNode", None]
    _ancestor_ids: Dict[bytes, int]
    _descendant_count: int
    _depth: int

    def __init__(self, model: ResourceModel, parent: Optional["ResourceNode"]):
        self.model = model
        self.parent = parent
        # Dict serves as an ordered set to preserve children insertion order
        self._children: Dict[ResourceNode, None] = dict()
        self._ancestor_ids: Dict[bytes, int] = dict()
        self._descendant_count = 0
        self._depth = 0
        if self.parent is not None:
            self.parent.add_child(self)
            self.model.parent_id = self.parent.model.id
        else:
            self.model.parent_id = None

    def add_child(self, child: "ResourceNode"):
        child._depth = self._depth + 1

        child._ancestor_ids = {
            parent_id: parent_depth + 1 for parent_id, parent_depth in self._ancestor_ids.items()
        }
        child._ancestor_ids[self.model.id] = 1

        parent: Optional[ResourceNode] = self
        while parent is not None:
            parent._descendant_count += child._descendant_count + 1
            parent = parent.parent

        self._children[child] = None

    def remove_child(self, child: "ResourceNode"):
        del self._children[child]
        parent: Optional[ResourceNode] = self
        while parent is not None:
            parent._descendant_count -= child._descendant_count + 1
            parent = parent.parent

        ids_to_clear = list(child._ancestor_ids.keys())

        def remove_ancestor_ids(descendent: "ResourceNode"):
            for ancestor_id in ids_to_clear:
                del descendent._ancestor_ids[ancestor_id]

            for _descendent in descendent._children.keys():
                remove_ancestor_ids(_descendent)

        remove_ancestor_ids(child)

    def has_ancestor(self, id: bytes, max_depth: int = -1, include_self: bool = False) -> bool:
        if include_self and id == self.model.id:
            return True
        ancestor_depth = self._ancestor_ids.get(id)
        if ancestor_depth is None:
            return False
        if max_depth < 0:
            return True
        return ancestor_depth <= max_depth

    def walk_ancestors(self, include_self: bool) -> Iterable["ResourceNode"]:
        if include_self:
            yield self
        parent = self.parent
        while parent is not None:
            yield parent
            parent = parent.parent

    def get_depth(self) -> int:
        return self._depth

    def get_descendant_count(self) -> int:
        return self._descendant_count

    def walk_descendants(
        self, include_self: bool, max_depth: int, _depth: int = 0
    ) -> Iterable["ResourceNode"]:
        if include_self:
            yield self
        if 0 <= max_depth <= _depth:
            return
        for child in self._children.keys():
            yield from child.walk_descendants(True, max_depth, _depth + 1)

    def __lt__(self, other):
        if not isinstance(other, ResourceNode):
            return NotImplemented
        return self.model.id < other.model.id

    def __eq__(self, other):
        if not isinstance(other, ResourceNode):
            return False
        return self.model.id == other.model.id

    def __hash__(self):
        return hash(self.model.id)


class ResourceAttributeIndex(Generic[T]):
    def __init__(self, attribute: ResourceIndexedAttribute[T]):
        self._attribute: ResourceIndexedAttribute[T] = attribute
        self.index: SortedList = SortedList()
        self.values_by_node_id: Dict[bytes, Any] = dict()

    def add_resource_attribute(
        self,
        value: T,
        resource: ResourceNode,
    ):
        if resource.model.id in self.values_by_node_id:
            if self.values_by_node_id[resource.model.id] != value:
                raise ValueError(
                    f"The provided resource {resource.model.id.hex()} is already in the "
                    f"index for {self._attribute.__name__} with a different value!"
                )
            else:
                return
        self.index.add((value, resource))
        self.values_by_node_id[resource.model.id] = value

    def remove_resource_attribute(
        self,
        resource: ResourceNode,
    ):
        if resource.model.id not in self.values_by_node_id:
            return
        value = self.values_by_node_id[resource.model.id]
        self.index.remove((value, resource))
        del self.values_by_node_id[resource.model.id]


class AttributeIndexDict(defaultdict):
    """
    `defaultdict` that passes the missing key to the default factory.

    See:
    <https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory>
    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class ResourceSortLogic(Generic[T], ABC):
    @abstractmethod
    def has_effect(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_match_count(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _get_attribute_value(self, resource: ResourceModel) -> Optional[T]:
        raise NotImplementedError()

    @abstractmethod
    def sort(self, resources: Iterable[ResourceModel]) -> Iterable[ResourceModel]:
        raise NotImplementedError()

    @abstractmethod
    def walk(self) -> Iterable[ResourceNode]:
        raise NotImplementedError()

    @abstractmethod
    def get_attribute(self) -> Optional[ResourceIndexedAttribute[T]]:
        raise NotImplementedError()

    @abstractmethod
    def get_direction(self) -> ResourceSortDirection:
        raise NotImplementedError()

    @staticmethod
    def create(
        r_sort: Optional[ResourceSort],
        attribute_indexes: Dict[ResourceIndexedAttribute[T], ResourceAttributeIndex],
    ) -> "ResourceSortLogic[T]":
        if r_sort is None:
            return NullResourceSortLogic()
        attribute_index = attribute_indexes[r_sort.attribute].index
        return ActiveResourceSortLogic[T](r_sort.attribute, attribute_index, r_sort.direction)


class ActiveResourceSortLogic(ResourceSortLogic[T]):
    def __init__(
        self,
        attribute: ResourceIndexedAttribute[T],
        index: List[Tuple[Any, ResourceNode]],
        direction: ResourceSortDirection = ResourceSortDirection.ASCENDANT,
    ):
        self.attribute: ResourceIndexedAttribute[T] = attribute  # type: ignore
        self.index: List[Tuple[Any, ResourceNode]] = index
        self.direction: ResourceSortDirection = direction

    def has_effect(self) -> bool:
        return True

    def get_match_count(self) -> int:
        return len(self.index)

    def get_attribute(self) -> ResourceIndexedAttribute[T]:
        return self.attribute

    def get_direction(self) -> ResourceSortDirection:
        return self.direction

    def _get_attribute_value(self, resource: ResourceModel) -> T:
        value = self.attribute.get_value(resource)
        if value is None:
            raise ValueError()
        else:
            return value

    def sort(self, resources: Iterable[ResourceModel]) -> Iterable[ResourceModel]:
        if self.attribute is None:
            raise ValueError("No attribute specified to sort on")
        reverse = self.direction != ResourceSortDirection.ASCENDANT
        return sorted(resources, key=self._get_attribute_value, reverse=reverse)

    def walk(self) -> Iterable[ResourceNode]:
        if self.index is None:
            raise ResourceServiceWalkError("Cannot walk a ResourceSortLogic with no index!")
        if self.direction is ResourceSortDirection.ASCENDANT:
            index = 0
            increment = 1
        else:
            index = len(self.index) - 1
            increment = -1
        max_index = len(self.index)
        while 0 <= index < max_index:
            yield self.index[index][1]
            index += increment


class NullResourceSortLogic(ResourceSortLogic):
    def has_effect(self) -> bool:
        return False

    def get_attribute(self) -> None:
        return None

    def get_match_count(self) -> int:  # pragma: no cover
        raise NotImplementedError()

    def _get_attribute_value(self, resource: ResourceModel) -> None:  # pragma: no cover
        raise NotImplementedError()

    def sort(
        self, resources: Iterable[ResourceModel]
    ) -> Iterable[ResourceModel]:  # pragma: no cover
        raise NotImplementedError()

    def walk(self) -> Iterable[ResourceNode]:  # pragma: no cover
        raise NotImplementedError()

    def get_direction(self) -> ResourceSortDirection:  # pragma: no cover
        raise NotImplementedError()  # pragma: no cover


class ResourceFilterLogic(Generic[T], ABC):
    def get_attribute(self) -> Optional[ResourceIndexedAttribute[T]]:
        return None

    @abstractmethod
    def filter(self, value: ResourceNode) -> bool:
        pass

    @abstractmethod
    def get_match_count(self) -> int:
        pass

    @abstractmethod
    def walk(self, direction: ResourceSortDirection) -> Iterable[ResourceNode]:
        pass

    @classmethod
    def get_attribute_value(
        cls, resource: ResourceNode, attribute_type: ResourceIndexedAttribute[T]
    ) -> Optional[T]:
        return attribute_type.get_value(resource.model)


class ResourceAttributeFilterLogic(ResourceFilterLogic[T], ABC):
    def __init__(
        self,
        attribute: ResourceIndexedAttribute[T],
        index: List[Tuple[T, ResourceNode]],
    ):
        self.attribute: ResourceIndexedAttribute[T] = attribute  # type: ignore
        self.index: List[Tuple[T, ResourceNode]] = index
        self._cached_ranges: Optional[Tuple[Range, ...]] = None

    def get_attribute(self) -> Optional[ResourceIndexedAttribute[T]]:
        return self.attribute

    @abstractmethod
    def _compute_ranges(self) -> Iterable[Range]:
        raise NotImplementedError()

    def walk(self, direction: ResourceSortDirection) -> Iterable[ResourceNode]:
        if self._cached_ranges is None:
            self._cached_ranges = tuple(sorted(self._compute_ranges(), key=lambda r: r.start))
        cached_ranges: Iterable[Range] = ()
        if direction is ResourceSortDirection.ASCENDANT:
            cached_ranges = self._cached_ranges
        else:
            cached_ranges = tuple(reversed(self._cached_ranges))

        for index_range in cached_ranges:
            if direction is ResourceSortDirection.ASCENDANT:
                index = index_range.start
                increment = 1
            else:
                index = index_range.end - 1
                increment = -1
            min_index = index_range.start
            max_index = index_range.end
            while min_index <= index < max_index:
                yield self.index[index][1]
                index += increment

    def get_match_count(self) -> int:
        if self._cached_ranges is None:
            self._cached_ranges = tuple(sorted(self._compute_ranges(), key=lambda r: r.start))
        return sum(r.length() for r in self._cached_ranges)


class ResourceAttributeRangeFilterLogic(ResourceAttributeFilterLogic, Generic[T]):
    def __init__(
        self,
        attribute: ResourceIndexedAttribute[T],
        index: List[Tuple[T, ResourceNode]],
        min: T = None,
        max: T = None,
    ):
        if min is None and max is None:
            raise ValueError("Invalid filter, either a min, a max or both must be provided")
        super().__init__(attribute, index)
        self.min: Optional[T] = min
        self.max: Optional[T] = max

    def _compute_ranges(self) -> Iterable[Range]:
        if self.min is not None:
            min_index = bisect.bisect_left(self.index, (self.min, LOW_VALUE))
        else:
            min_index = 0
        if self.max is not None:
            # TODO: There should most likely be a +1 in here
            max_index = bisect.bisect_left(self.index, (self.max, LOW_VALUE))
        else:
            max_index = len(self.index)
        return (Range(min_index, max_index),)

    def filter(self, resource: ResourceNode) -> bool:
        value = self.get_attribute_value(resource, self.attribute)
        if value is None:
            return False
        if self.min is not None and self.max is not None:
            return self.min <= value < self.max
        if self.min is not None:
            return self.min <= value
        elif self.max is not None:
            return value < self.max
        else:
            raise ValueError("Invalid filter, either a min, a max or both must be provided")


class ResourceAttributeValueFilterLogic(ResourceAttributeFilterLogic, Generic[T]):
    def __init__(
        self,
        attribute: ResourceIndexedAttribute[T],
        index: List[Tuple[T, ResourceNode]],
        value: T,
    ):
        super().__init__(attribute, index)
        self.value: T = value

    def _compute_ranges(self) -> Iterable[Range]:
        return (
            Range(
                bisect.bisect_left(self.index, (self.value, LOW_VALUE)),
                bisect.bisect_right(self.index, (self.value, HIGH_VALUE)),
            ),
        )

    def filter(self, resource: ResourceNode) -> bool:
        value = self.get_attribute_value(resource, self.attribute)
        if value is None:
            return False
        return value == self.value


class ResourceAttributeValuesFilterLogic(ResourceAttributeFilterLogic, Generic[T]):
    def __init__(
        self,
        attribute: ResourceIndexedAttribute[T],
        index: List[Tuple[T, ResourceNode]],
        values: Tuple[T, ...],
    ):
        super().__init__(attribute, index)
        self.index = index
        self.values: Set[T] = set(values)

    def _compute_ranges(self) -> Iterable[Range]:
        for value in self.values:
            yield Range(
                bisect.bisect_left(self.index, (value, LOW_VALUE)),
                bisect.bisect_right(self.index, (value, HIGH_VALUE)),
            )

    def filter(self, resource: ResourceNode) -> bool:
        value = self.get_attribute_value(resource, self.attribute)
        if value is None:
            return False
        return value in self.values


class ResourceTagOrFilterLogic(ResourceFilterLogic):
    def __init__(
        self,
        indexes: Dict[ResourceTag, Set[ResourceNode]],
        tags: Tuple[ResourceTag, ...],
    ):
        if len(tags) == 0:
            raise ValueError(
                "Cannot instantiate the ResourceTagOrFilterLogic class with an empty set of tags "
                "to filter on."
            )
        self.indexes = indexes
        self.tags = tags

    def filter(self, resource: ResourceNode) -> bool:
        for tag in self.tags:
            if resource.model.has_tag(tag):
                return True
        return False

    def get_match_count(self) -> int:
        count = 0
        for tag in self.tags:
            count += len(self.indexes[tag])
        return count

    def walk(self, direction: ResourceSortDirection) -> Iterable[ResourceNode]:
        processed_ids = set()
        for tag in self.tags:
            for resource in self.indexes[tag]:
                resource_m = resource.model
                if resource_m.id in processed_ids:
                    continue
                processed_ids.add(resource_m.id)
                yield resource


class ResourceTagAndFilterLogic(ResourceFilterLogic):
    def __init__(
        self,
        indexes: Dict[ResourceTag, Set[ResourceNode]],
        tags: Tuple[ResourceTag, ...],
    ):
        if len(tags) == 0:
            raise ValueError(
                "Cannot instantiate the ResourceTagAndFilterLogic class with an empty set of tags "
                "to filter on."
            )
        self.indexes: Dict[ResourceTag, Set[ResourceNode]] = indexes
        self.tags = tags

        self._walk_tag: Optional[ResourceTag] = None
        self._filter_tags: Optional[Tuple[ResourceTag, ...]] = None

    def filter(self, resource: ResourceNode) -> bool:
        for tag in self.tags:
            if not resource.model.has_tag(tag):
                return False
        return True

    def _compute_tags(self) -> Tuple[ResourceTag, Optional[Tuple[ResourceTag, ...]]]:
        if self._walk_tag is not None:
            return self._walk_tag, self._filter_tags
        min_size = sys.maxsize
        walk_tag = None
        for tag in self.tags:
            index_size = len(self.indexes[tag])
            if index_size < min_size:
                walk_tag = tag
                min_size = index_size
        if walk_tag is None:
            # No tags in self.tags had fewer than sys.maxsize resources with that tag. Choose one
            # arbitrarily to be the walk tag then, since they all have equal (very high) cost.
            walk_tag = self.tags[0]
        self._walk_tag = walk_tag
        filter_tags = tuple(filter(lambda t: t != walk_tag, self.tags))
        self._filter_tags = filter_tags

        return walk_tag, filter_tags

    def get_match_count(self) -> int:
        walk_tag, _ = self._compute_tags()
        return len(self.indexes[walk_tag])

    def walk(self, direction: ResourceSortDirection) -> Iterable[ResourceNode]:
        walk_tag, filter_tags = self._compute_tags()
        main_index = self.indexes[walk_tag]
        for resource in main_index:
            if filter_tags is not None and len(filter_tags) > 0:
                for tag in filter_tags:
                    if resource in self.indexes[tag]:
                        yield resource
            else:
                yield resource


class ResourceAncestorFilterLogic(ResourceFilterLogic):
    def __init__(
        self,
        root: ResourceNode,
        include_root: bool = False,
        max_depth: int = -1,
    ):
        self.root = root
        self.include_root = include_root
        self.max_depth = max_depth

    def filter(self, resource: ResourceNode) -> bool:
        return resource.has_ancestor(self.root.model.id, self.max_depth, self.include_root)

    def get_match_count(self) -> int:
        count = self.root.get_descendant_count()
        if self.include_root:
            count += 1
        return count

    def walk(self, direction: ResourceSortDirection) -> Iterable[ResourceNode]:
        return self.root.walk_descendants(include_self=self.include_root, max_depth=self.max_depth)


class AggregateResourceFilterLogic:
    def __init__(self, filters: Tuple[ResourceFilterLogic, ...]):
        self.filters = filters

    @staticmethod
    def _create_attribute_filter(
        attribute_filter: ResourceAttributeFilter, attribute_index: List[Tuple[T, ResourceNode]]
    ) -> ResourceFilterLogic[T]:
        if isinstance(attribute_filter, ResourceAttributeRangeFilter):
            return ResourceAttributeRangeFilterLogic(
                attribute_filter.attribute,
                attribute_index,
                attribute_filter.min,
                attribute_filter.max,
            )
        elif isinstance(attribute_filter, ResourceAttributeValueFilter):
            return ResourceAttributeValueFilterLogic(
                attribute_filter.attribute, attribute_index, attribute_filter.value
            )
        elif isinstance(attribute_filter, ResourceAttributeValuesFilter):
            return ResourceAttributeValuesFilterLogic(
                attribute_filter.attribute, attribute_index, attribute_filter.values
            )
        else:
            raise ValueError(f"Unknown filter of type {type(attribute_filter).__name__}")

    @staticmethod
    def create(
        r_filter: Optional[ResourceFilter],
        tag_indexes: Dict[ResourceTag, Set[ResourceNode]],
        attribute_indexes: Dict[ResourceIndexedAttribute[T], ResourceAttributeIndex[T]],
        ancestor: ResourceNode = None,
        max_depth: int = -1,
    ) -> "AggregateResourceFilterLogic":
        filters: List[ResourceFilterLogic] = []
        if r_filter is not None:
            if r_filter.tags is not None:
                if r_filter.tags_condition is ResourceFilterCondition.AND:
                    filters.append(ResourceTagAndFilterLogic(tag_indexes, tuple(r_filter.tags)))
                else:
                    filters.append(ResourceTagOrFilterLogic(tag_indexes, tuple(r_filter.tags)))
            if r_filter.attribute_filters is not None:
                for attribute_filter in r_filter.attribute_filters:
                    filters.append(
                        AggregateResourceFilterLogic._create_attribute_filter(
                            attribute_filter, attribute_indexes[attribute_filter.attribute].index
                        )
                    )

            include_root = r_filter.include_self
        else:
            include_root = False
        if ancestor is not None:
            filters.append(ResourceAncestorFilterLogic(ancestor, include_root, max_depth))
        return AggregateResourceFilterLogic(tuple(filters))

    def ignore_filter(self, filter_logic: ResourceFilterLogic):
        if filter_logic is None:
            raise ValueError("Invalid index filter logic")
        self.filters = tuple(ix for ix in self.filters if ix != filter_logic)

    def has_effect(self) -> bool:
        return len(self.filters) > 0

    def filter(self, resource: ResourceNode) -> bool:
        for filter_logic in self.filters:
            if not filter_logic.filter(resource):
                return False
        return True


def _type_to_str(t: type) -> str:
    return f"{inspect.getmodule(t).__name__}.{t.__name__}"


def _type_from_str(t: str) -> type:
    # Taken from type_serializer.py in the serialization service
    module_path, cls_name = t.rsplit(".", maxsplit=1)
    module = sys.modules[module_path]
    return getattr(module, cls_name)


# TODO: Type hint
def _serialize_attribute_field(attr):
    if any(
        [
            isinstance(attr, str),
            isinstance(attr, int),
            isinstance(attr, float),
            isinstance(attr, bytes),
            attr is None,
        ]
    ):
        return attr
    return pickle.dumps(attr)


# TODO: Type hint
def _deserialize_attribute_field(attributes_type, name, data):
    if any(
        [
            isinstance(data, str),
            isinstance(data, int),
            isinstance(data, float),
            data is None,
        ]
    ):
        return data
    data_type = next(
        field.type for field in dataclasses.fields(attributes_type) if field.name == name
    )
    if data_type == bytes:
        return data
    return pickle.loads(data)


def _get_model_by_id(
    resource_id: bytes, conn: sqlite3.Connection, force_fetch=False
) -> ResourceModel:
    if not force_fetch:
        (resource_model_data,) = conn.execute(
            "SELECT model FROM resources WHERE resources.resource_id = ? LIMIT 1", (resource_id,)
        ).fetchone()
        return pickle.loads(resource_model_data)
    ids = conn.execute(
        """SELECT data_id, parent_id 
        FROM resources 
        WHERE resources.resource_id = ?""",
        (resource_id,),
    ).fetchone()
    if ids is None:
        raise NotFoundError(f"Resource ID {resource_id.hex()} does not exist")
    data_id, parent_id = ids
    tags = set()
    for (tag_name,) in conn.execute(
        """SELECT tag 
        FROM tags 
        WHERE tags.resource_id = ?""",
        (resource_id,),
    ):
        tags.add(_type_from_str(tag_name))
    _attributes = defaultdict(dict)
    for (attributes_type, field_name, field_value) in conn.execute(
        """SELECT attributes_type, field_name, field_value 
        FROM attributes 
        WHERE attributes.resource_id = ?""",
        (resource_id,),
    ):
        _attributes[_type_from_str(attributes_type)][field_name] = _deserialize_attribute_field(
            _type_from_str(attributes_type), field_name, field_value
        )
    attributes = {
        attributes_type: attributes_type(**fields)
        for attributes_type, fields in _attributes.items()
    }
    data_dependencies = defaultdict(set)
    for (
        dependent_resource_id,
        component_id,
        attributes_type,
        range_start,
        range_end,
    ) in conn.execute(
        """SELECT dependent_resource_id, component_id, attributes_type, range_start, range_end 
        FROM data_dependencies 
        WHERE data_dependencies.resource_id = ?""",
        (resource_id,),
    ):
        r = Range(range_start, range_end)
        # TODO: Fix typing
        attr_dep = ResourceAttributeDependency(
            dependent_resource_id, component_id, _type_from_str(attributes_type)
        )
        data_dependencies[attr_dep].add(r)
    attribute_dependencies = defaultdict(set)
    for (attributes_type, dependent_resource_id, component_id) in conn.execute(
        """SELECT attributes_type, dependent_resource_id, component_id 
        FROM attribute_dependencies 
        WHERE attribute_dependencies.resource_id = ?""",
        (resource_id,),
    ):
        # TODO: Fix typing
        deserialized_type = _type_from_str(attributes_type)
        attr_dep = ResourceAttributeDependency(
            dependent_resource_id, component_id, deserialized_type
        )
        attribute_dependencies[deserialized_type] = attr_dep
    component_versions = dict(
        conn.execute(
            """SELECT component_id, component_version 
            FROM component_versions 
            WHERE component_versions.resource_id = ?""",
            (resource_id,),
        )
    )
    components_by_attributes = {
        _type_from_str(attributes_type): (component_id, component_version)
        for attributes_type, component_id, component_version in conn.execute(
            """SELECT attributes_type, component_id, version 
            FROM components_by_attributes 
            WHERE components_by_attributes.resource_id = ?""",
            (resource_id,),
        )
    }
    result = ResourceModel(
        id=resource_id,
        data_id=data_id,
        parent_id=parent_id,
        tags=tags,
        attributes=attributes,
        data_dependencies=dict(data_dependencies),
        attribute_dependencies=dict(attribute_dependencies),
        component_versions=component_versions,
        components_by_attributes=components_by_attributes,
    )
    conn.execute(
        "UPDATE resources SET model = ? WHERE resources.resource_id = ?",
        (pickle.dumps(result), resource_id),
    )
    return result


def _delete_resources(
    resource_ids: Iterable[bytes], conn: sqlite3.Connection
) -> Iterable[ResourceModel]:
    result = [
        _get_model_by_id(descendant_id, conn)
        for resource_id in resource_ids
        for (descendant_id,) in conn.execute(
            """SELECT descendant_id
            FROM closure
            WHERE closure.ancestor_id = ?""",
            (resource_id,),
        )
    ]
    conn.executemany(
        """DELETE FROM closure
        WHERE closure.ancestor_id = ?""",
        [(resource_id,) for resource_id in resource_ids],
    )
    # TODO: Delete from everywhere else
    return result


def _get_descendants(
    resource_id: bytes,
    conn: sqlite3.Connection,
    max_count: int = -1,
    max_depth: int = -1,
    r_filter: Optional[ResourceFilter] = None,
    r_sort: Optional[ResourceSort] = None,
) -> Iterable[ResourceModel]:
    filter_query_parameters, filter_query_list = [], []
    if r_filter:
        # TODO
        # attribute_filters: Optional[Iterable[ResourceAttributeFilter]] = None
        if r_filter.tags:
            tag_list = list(r_filter.tags)
            condition = "closure.descendant_id IN (SELECT resource_id FROM tags WHERE "
            condition_join = (
                " AND " if r_filter.tags_condition == ResourceFilterCondition.AND else " OR "
            )
            condition += condition_join.join([f"tags.tag = ?"] * len(tag_list))
            condition += ")"
            filter_query_list.append(condition)
            filter_query_parameters.extend(map(_type_to_str, tag_list))
    if max_depth != -1:
        filter_query_list.append("depth <= ?")
        filter_query_parameters.append(max_depth)
    if max_count != -1:
        filter_query_parameters.append(max_count)
    return [
        _get_model_by_id(descendant_id, conn)
        for (descendant_id,) in conn.execute(
            f"""SELECT descendant_id
            FROM closure
            WHERE ancestor_id = ?
            {('AND ' + ' AND '.join(filter_query_list)) if filter_query_list else ''}
            ORDER BY depth DESC
            {'LIMIT ?' if max_count != -1 else ''}""",
            (resource_id, *filter_query_parameters),
        )
    ]


def _update(diff: ResourceModelDiff, conn: sqlite3.Connection) -> ResourceModel:
    resource_id = diff.id
    conn.executemany(
        "INSERT INTO tags VALUES(?, ?)",
        [(resource_id, _type_to_str(tag)) for tag in diff.tags_added],
    )
    conn.executemany(
        """DELETE FROM tags 
        WHERE tags.resource_id = ? 
        AND tags.tag = ?""",
        [(resource_id, _type_to_str(tag)) for tag in diff.tags_removed],
    )
    conn.executemany(
        "INSERT INTO attributes VALUES(?, ?, ?, ?)",
        [
            (resource_id, _type_to_str(attributes_type), name, _serialize_attribute_field(value))
            for attributes_type, attribute in diff.attributes_added.items()
            for name, value in dataclasses.asdict(attribute).items()
        ],
    )
    conn.executemany(
        """DELETE FROM attributes 
        WHERE attributes.resource_id = ? 
        AND attributes.attributes_type = ?""",
        [
            (resource_id, _type_to_str(attributes_type))
            for attributes_type in diff.attributes_removed
        ],
    )
    conn.executemany(
        "INSERT INTO data_dependencies VALUES(?, ?, ?, ?, ?, ?)",
        [
            (
                resource_id,
                dep.dependent_resource_id,
                dep.component_id,
                _type_to_str(dep.attributes),
                data_range.start,
                data_range.end,
            )
            for dep, data_range in diff.data_dependencies_added
        ],
    )
    conn.executemany(
        """DELETE FROM data_dependencies 
        WHERE data_dependencies.resource_id = ? 
        AND data_dependencies.dependent_resource_id = ? 
        AND data_dependencies.component_id = ? 
        AND data_dependencies.attributes_type = ?""",
        [
            (resource_id, dep.dependent_resource_id, dep.component_id, _type_to_str(dep.attributes))
            for dep in diff.data_dependencies_removed
        ],
    )
    conn.executemany(
        "INSERT INTO attribute_dependencies VALUES(?, ?, ?, ?)",
        [
            (
                resource_id,
                _type_to_str(attributes_type),
                dep.dependent_resource_id,
                dep.component_id,
            )
            for attributes_type, dep in diff.attribute_dependencies_added
        ],
    )
    conn.executemany(
        """DELETE FROM attribute_dependencies 
        WHERE attribute_dependencies.resource_id = ? 
        AND attribute_dependencies.attributes_type = ? 
        AND attribute_dependencies.dependent_resource_id = ? 
        AND attribute_dependencies.component_id = ?""",
        [
            (
                resource_id,
                _type_to_str(attributes_type),
                dep.dependent_resource_id,
                dep.component_id,
            )
            for attributes_type, dep in diff.attribute_dependencies_removed
        ],
    )
    conn.executemany(
        "INSERT INTO component_versions VALUES(?, ?, ?)",
        [
            (resource_id, component_id, component_version)
            for component_id, component_version in diff.component_versions_added
        ],
    )
    conn.executemany(
        """DELETE FROM component_versions 
        WHERE component_versions.resource_id = ? 
        AND component_versions.component_id = ?""",
        [(resource_id, component_id) for component_id in diff.component_versions_removed],
    )
    conn.executemany(
        "INSERT INTO components_by_attributes VALUES(?, ?, ?, ?)",
        [
            (resource_id, _type_to_str(attributes_type), component_id, component_version)
            for (
                attributes_type,
                component_id,
                component_version,
            ) in diff.attributes_component_added
        ],
    )
    conn.executemany(
        """DELETE FROM components_by_attributes 
        WHERE components_by_attributes.resource_id = ? 
        AND components_by_attributes.attributes_type""",
        [
            (resource_id, _type_to_str(attributes_type))
            for attributes_type in diff.attributes_component_removed
        ],
    )
    return _get_model_by_id(resource_id, conn, force_fetch=True)


class ResourceService(ResourceServiceInterface):
    def __init__(self):
        self._conn = sqlite3.connect(
            ":memory:",  # detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        with self._conn as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = normal;")
            conn.execute("PRAGMA temp_store = memory;")
            conn.execute("PRAGMA mmap_size = 30000000000;")

            conn.execute(
                """CREATE TABLE resources (
                    resource_id PRIMARY KEY, 
                    data_id, 
                    parent_id,
                    model
                )"""
            )
            conn.execute(
                """CREATE TABLE tags (
                    resource_id, 
                    tag, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE attributes (
                    resource_id, 
                    attributes_type, 
                    field_name, 
                    field_value, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE data_dependencies (
                    resource_id, 
                    dependent_resource_id, 
                    component_id, 
                    attributes_type, 
                    range_start, 
                    range_end, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE attribute_dependencies (
                    resource_id, 
                    attributes_type, 
                    dependent_resource_id, 
                    component_id, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE component_versions (
                    resource_id, 
                    component_id, 
                    component_version, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE components_by_attributes (
                    resource_id, 
                    attributes_type, 
                    component_id, 
                    version, 
                    FOREIGN KEY (resource_id) REFERENCES resources (resource_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE closure (
                    ancestor_id,
                    descendant_id,
                    depth,
                    FOREIGN KEY (ancestor_id) REFERENCES resources (resource_id),
                    FOREIGN KEY (descendant_id) REFERENCES resources (resource_id)
                )"""
            )

    async def create(self, resource: ResourceModel) -> ResourceModel:
        with self._conn as conn:
            if conn.execute(
                "SELECT resource_id FROM resources WHERE resource_id = ?", (resource.id,)
            ).fetchone():
                raise AlreadyExistError(f"Resource {resource.id.hex()} already exists")
            conn.execute(
                "INSERT INTO resources VALUES(?, ?, ?, ?)",
                (resource.id, resource.data_id, resource.parent_id, pickle.dumps(resource)),
            )
            conn.executemany(
                "INSERT INTO tags VALUES(?, ?)",
                [(resource.id, _type_to_str(tag)) for tag in resource.tags],
            )
            conn.executemany(
                "INSERT INTO attributes VALUES(?, ?, ?, ?)",
                [
                    (
                        resource.id,
                        _type_to_str(attributes_type),
                        name,
                        _serialize_attribute_field(value),
                    )
                    for attributes_type, attribute in resource.attributes.items()
                    for name, value in dataclasses.asdict(attribute).items()
                ],
            )
            conn.executemany(
                "INSERT INTO data_dependencies VALUES(?, ?, ?, ?, ?, ?)",
                [
                    (
                        resource.id,
                        dep.dependent_resource_id,
                        dep.component_id,
                        _type_to_str(dep.attributes),
                        data_range.start,
                        data_range.end,
                    )
                    for dep, ranges in resource.data_dependencies.items()
                    for data_range in ranges
                ],
            )
            conn.executemany(
                "INSERT INTO attribute_dependencies VALUES(?, ?, ?, ?)",
                [
                    (
                        resource.id,
                        _type_to_str(attributes_type),
                        dep.dependent_resource_id,
                        dep.component_id,
                    )
                    for attributes_type, attribute_dependencies in resource.attribute_dependencies.items()
                    for dep in attribute_dependencies
                ],
            )
            conn.executemany(
                "INSERT INTO component_versions VALUES(?, ?, ?)",
                [
                    (resource.id, component_id, component_version)
                    for component_id, component_version in resource.component_versions.items()
                ],
            )
            conn.executemany(
                "INSERT INTO components_by_attributes VALUES(?, ?, ?, ?)",
                [
                    (resource.id, _type_to_str(attributes_type), component_id, component_version)
                    for attributes_type, (
                        component_id,
                        component_version,
                    ) in resource.components_by_attributes.items()
                ],
            )
            conn.executemany(
                "INSERT INTO closure VALUES (?, ?, ?)",
                [
                    (ancestor_id, resource.id, depth + 1)
                    for (ancestor_id, depth) in conn.execute(
                        """SELECT ancestor_id, depth FROM closure WHERE closure.descendant_id = ?""",
                        (resource.parent_id,),
                    )
                ]
                + (
                    [(resource.parent_id, resource.id, 1)] if resource.parent_id is not None else []
                ),
            )
        return resource

    async def get_root_resources(self) -> Iterable[ResourceModel]:
        with self._conn as conn:
            return [
                _get_model_by_id(root_id, conn)
                for (root_id,) in conn.execute(
                    "SELECT resource_id FROM resources WHERE resources.parent_id IS NULL"
                )
            ]

    async def verify_ids_exist(self, resource_ids: Iterable[bytes]) -> Iterable[bool]:
        with self._conn as conn:
            return [
                (
                    True
                    if conn.execute(
                        "SELECT resource_id FROM resources WHERE resources.resource_id = ?",
                        (resource_id,),
                    )
                    else False
                )
                for resource_id in resource_ids
            ]

    async def get_by_data_ids(self, data_ids: Iterable[bytes]) -> Iterable[ResourceModel]:
        try:
            with self._conn as conn:
                resource_ids = [
                    conn.execute(
                        "SELECT resource_id FROM resources WHERE resources.data_id = ?",
                        (data_id,),
                    ).fetchone()[0]
                    for data_id in data_ids
                ]
        except TypeError:
            raise NotFoundError()
        return await self.get_by_ids(resource_ids)

    async def get_by_ids(self, resource_ids: Iterable[bytes]) -> Iterable[ResourceModel]:
        # TODO: Optimize
        with self._conn as conn:
            return [_get_model_by_id(resource_id, conn) for resource_id in resource_ids]

    async def get_by_id(self, resource_id: bytes) -> ResourceModel:
        with self._conn as conn:
            return _get_model_by_id(resource_id, conn)

    async def get_depths(self, resource_ids: Iterable[bytes]) -> Iterable[int]:
        return [
            len(list(await self.get_ancestors_by_id(resource_id))) for resource_id in resource_ids
        ]

    # @lru_cache
    async def get_ancestors_by_id(
        self, resource_id: bytes, max_count: int = -1, r_filter: Optional[ResourceFilter] = None
    ) -> Iterable[ResourceModel]:
        filter_query_list, filter_query_parameters = [], []
        if r_filter:
            # TODO
            # include_self: bool = False
            # attribute_filters: Optional[Iterable[ResourceAttributeFilter]] = None
            # max_count
            if r_filter.tags:
                tag_list = list(r_filter.tags)
                condition = "ancestor_id IN (SELECT resource_id FROM tags WHERE "
                condition_join = (
                    " AND " if r_filter.tags_condition == ResourceFilterCondition.AND else " OR "
                )
                condition += condition_join.join([f"tags.tag = ?"] * len(tag_list))
                condition += ")"
                filter_query_list.append(condition)
                filter_query_parameters.extend(map(_type_to_str, tag_list))
        if max_count != -1:
            filter_query_parameters.append(max_count)
        with self._conn as conn:
            result = [
                _get_model_by_id(ancestor_id, conn)
                for (ancestor_id,) in conn.execute(
                    f"""SELECT ancestor_id 
                    FROM closure 
                    WHERE descendant_id = ?
                    {('AND (' + ' AND '.join(filter_query_list) + ')') if filter_query_list else ''}
                    {'LIMIT ?' if max_count != -1 else ''}""",
                    (resource_id, *filter_query_parameters),
                )
            ]
            return result

    async def get_descendants_by_id(
        self,
        resource_id: bytes,
        max_count: int = -1,
        max_depth: int = -1,
        r_filter: Optional[ResourceFilter] = None,
        r_sort: Optional[ResourceSort] = None,
    ) -> Iterable[ResourceModel]:
        with self._conn as conn:
            return _get_descendants(resource_id, conn, max_count, max_depth, r_filter, r_sort)

    async def get_siblings_by_id(
        self,
        resource_id: bytes,
        max_count: int = -1,
        r_filter: Optional[ResourceFilter] = None,
        r_sort: Optional[ResourceSort] = None,
    ) -> Iterable[ResourceModel]:
        if max_count != -1 or r_sort:
            # TODO
            raise NotImplementedError()
        filter_query_parameters, filter_query_list = [], []
        if r_filter:
            # TODO
            # include_self: bool = False
            # attribute_filters: Optional[Iterable[ResourceAttributeFilter]] = None
            if r_filter.tags:
                tag_list = list(r_filter.tags)
                condition = "resources.resource_id IN (SELECT resource_id FROM tags WHERE "
                condition_join = (
                    " AND " if r_filter.tags_condition == ResourceFilterCondition.AND else " OR "
                )
                condition += condition_join.join([f"tags.tag = ?"] * len(tag_list))
                condition += ")"
                filter_query_list.append(condition)
                filter_query_parameters.extend(map(_type_to_str, tag_list))
        with self._conn as conn:
            # TODO: Collapse into one query
            (parent_id,) = conn.execute(
                """SELECT parent_id 
                FROM resources 
                WHERE resources.resource_id = ?""",
                (resource_id,),
            ).fetchone()
            return [
                _get_model_by_id(sibling_id, conn)
                for (sibling_id,) in conn.execute(
                    f"""SELECT resource_id 
                    FROM resources 
                    WHERE resources.parent_id = ? 
                    AND resources.resource_id != ?
                    {('AND ' + ' AND '.join(filter_query_list)) if filter_query_list else ''}""",
                    (parent_id, resource_id),
                )
            ]

    async def update(self, resource_diff: ResourceModelDiff) -> ResourceModel:
        with self._conn as conn:
            return _update(resource_diff, conn)

    async def update_many(
        self, resource_diffs: Iterable[ResourceModelDiff]
    ) -> Iterable[ResourceModel]:
        with self._conn as conn:
            return [_update(resource_diff, conn) for resource_diff in resource_diffs]

    async def rebase_resource(self, resource_id: bytes, new_parent_id: bytes):
        with self._conn as conn:
            conn.execute(
                "UPDATE resources SET parent_id = ? WHERE resources.resource_id = ?",
                (new_parent_id, resource_id),
            )
            conn.execute("DELETE FROM closure WHERE closure.descendant_id = ?", (resource_id,))
            conn.executemany(
                "INSERT INTO closure VALUES (?, ?, ?)",
                [
                    (ancestor_id, resource_id, depth + 1)
                    for (ancestor_id, depth) in conn.execute(
                        """SELECT ancestor_id, depth 
                    FROM closure 
                    WHERE closure.descendant_id = ?""",
                        (new_parent_id,),
                    )
                ]
                + ([(new_parent_id, resource_id, 1)] if new_parent_id is not None else []),
            )

    async def delete_resource(self, resource_id: bytes) -> Iterable[ResourceModel]:
        return await self.delete_resources([resource_id])

    async def delete_resources(self, resource_ids: Iterable[bytes]) -> Iterable[ResourceModel]:
        with self._conn as conn:
            return _delete_resources(resource_ids, conn)
