from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union, List

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    pos = 0
    for i in range(len(strides)):
        pos += index[i] * strides[i]
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # contiguous_strides = np.zeros(len(shape), dtype=np.float64)
    # contiguous_strides[-1] = 1
    # for i in range(len(shape) - 1, 0, -1):
    #     contiguous_strides[i] = shape[i] * contiguous_strides[i + 1]
    # for i in range(len(shape)):
    #     out_index[i] = ordinal // contiguous_strides[i]
    #     ordinal = ordinal % contiguous_strides[i]

    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Union[Index, List[int]],
    big_shape: Shape,
    shape: Shape,
    out_index: Union[OutIndex, List[int]],
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + len(big_shape) - len(shape)]
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # if len(shape1) > len(shape2):
    #     nshape2 = np.ones(len(shape1), dtype=np.int32)
    #     for i in range(len(shape1) - len(shape2), len(shape1)):
    #         nshape2[i] = shape2[i]
    #     shape2 = nshape2
    # else:
    #     nshape1 = np.ones(len(shape2), dtype=np.int32)
    #     for i in range(len(shape2) - len(shape1), len(shape2)):
    #         nshape1[i] = shape1[i]
    #     shape1 = nshape1
    # res = np.ones(len(shape1), dtype=np.int32)
    # for i in range(len(shape1)):
    #     if shape1[i] == shape2[i]:
    #         res[i] = shape1[i]
    #     elif shape1[i] == 1:
    #         res[i] = shape2[i]
    #     elif shape2[i] == 1:
    #         res[i] = shape1[i]
    #     else:
    #         raise IndexingError
    # return tuple(res)

    a, b = shape1, shape2
    m = len(a) if len(a) > len(b) else len(b)
    c_rev = np.zeros(m, dtype=int)
    a_rev = a[::-1]
    b_rev = b[::-1]
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
    return tuple(c_rev[::-1])


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(list(shape)))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape."""
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Indexes into a position of the tensor"""
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Returns a generator to iterate over every index of the tensor"""
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Gets the value at a specific key in the tensor"""
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value at a specific key in the tensor"""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = [0 for _ in range(len(self.shape))]
        new_strides = [0 for _ in range(len(self.shape))]
        for i in range(len(self.shape)):
            new_shape[i] = self.shape[order[i]]
            new_strides[i] = self.strides[order[i]]
        return TensorData(self._storage, tuple(new_shape), tuple(new_strides))

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
