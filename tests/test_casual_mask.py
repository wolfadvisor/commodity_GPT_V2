import torch
from commodity_gpt_v2.attention.casual_mask import CasualMask

def test_mask_shape():

    seq_len = 4

    mask = CasualMask.create(
        seq_len,
        torch.device("cpu")
    )


    assert mask.shape == (4,4)

    def test_mask_dtype():

        mask = CasualMask.create(4, torch.device("cpu"))

        assert mask.dtype == torch.bool

    def test_mask_values():

        expected = torch.tensor(
             [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
        )

        mask = CasualMask.create(4,torch.device('cpu'))
        assert torch.equal(mask, expected)

        def test_large_mask():
            mask = CasualMask.create(
                128,
                torch.device('cpu')
            )

            assert mask.shape == (128,128)

    def test_print_mask():

        mask = CasualMask.create(
            8,
            torch.device("cpu")
        )

        print(mask)

