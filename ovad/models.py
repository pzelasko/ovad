from typing import Optional

from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from snowfall.models import AcousticModel
from snowfall.training.diagnostics import measure_weight_norms


class TdnnLstm1a(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=512, hidden_size=512, num_layers=1) for _ in range(5)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=512, affine=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(
        self, tb_writer: SummaryWriter, global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            "train/weight_l2_norms",
            measure_weight_norms(self, norm="l2"),
            global_step=global_step,
        )
        tb_writer.add_scalars(
            "train/weight_max_norms",
            measure_weight_norms(self, norm="linf"),
            global_step=global_step,
        )
