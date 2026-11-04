"""Public tuple overload vectors from blake3 1.0.8 and hashlib SHA256."""

import struct

import pytest
import torch

from fss_crypto import hash


_SEED = (0x12345678, 0x23456789, 0x3456789a, 0x456789aa)
_VECTORS = {
    'blake3': (
        ((0, 0, 0, 0), "4f03bce401a5da177f0534476c196af84428ec9014fd02fae81f10e9e1373ff6b52cf809ec0d917f56dd890f38aa9b4940076e930043aa217d382274620527bd"),
        ((0, 0, 0, 1), "2fcffdb333d9d874217216982452c2833b1b8dd76fac7efc262936850023d080c5478b682f7d29cefae345024a06620d0766805d7527d6035c7550a8ae66178a"),
        ((1126240237, 1412567294, 1698898191, 1985229328), "e7e55ddc70bacc8d03ac1918161c025be0e745542e7a934e6ee63d4e019b3e7f2529e032a81ab87b03db05e03a092451dec40de4e28a848aef2fec97562abdae"),
    ),
    'sha256': (
        ((0, 0, 0, 0), "794a69a0b7613d486c7197b4b80d4c17937eb02ff12620c61e70d0904e46cdbfb07bfdd015e9779940298f34d1837deeb2ac3f397aa7cb8817ff232a2c68dc92"),
        ((0, 0, 0, 1), "7b61d629abb86a217c9f726cc897371f6ab03f086b106a920639ef85a35f34c4560be8f564858902ed9b3258572b3b123b9470e6f18feef63cde18141b227e38"),
        ((1126240237, 1412567294, 1698898191, 1985229328), "522ef1a817a57ad26ad03cf12126f77a71a994ad597e24a5d397afcf38d09af25de4af1f07355539ae26581cb1e02d1587d9ac3dd5b14357f15c61e26ebb4003"),
    ),
}


@pytest.mark.parametrize("backend", ("blake3", "sha256"))
def test_tuple_hash_coordinate_seed_golden(backend):
    hasher = (hash.Blake3(bytes(range(32))) if backend == "blake3"
              else hash.Sha256(bytes(range(16))))
    for coordinate, expected in _VECTORS[backend]:
        for reserved_bit in (0, 1):
            seed = (*_SEED[:3], _SEED[3] | reserved_bit)
            message = torch.tensor((coordinate, seed), dtype=torch.int32)
            original = message.clone()
            output = hasher.hash(message)
            assert output.dtype == torch.int32 and output.shape == (4, 4)
            actual = struct.pack("<16I", *(word & 0xffffffff
                                         for row in output.tolist() for word in row))
            assert actual == bytes.fromhex(expected)
            assert torch.equal(message, original)
