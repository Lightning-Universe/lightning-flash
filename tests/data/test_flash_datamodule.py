from flash.data.data_module import DataModule


def test_flash_special_arguments(tmpdir):

    class CustomDataModule(DataModule):

        test = 1

    dm = CustomDataModule()
    CustomDataModule.test = 2
    assert dm.test == 2

    class CustomDataModule2(DataModule):

        test = 1
        __flash_special_attr__ = ["test"]

    dm = CustomDataModule2()
    CustomDataModule2.test = 2
    assert dm.test == 1
