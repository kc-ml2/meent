import torch


class Compress(torch.autograd.Function):

    @staticmethod
    def setup_context(ctx, inputs, output):
        # layer_info, datatype = inputs
        pass

    @staticmethod
    def forward(col_list):

        x_list = torch.ones((len(col_list), 1))
        for i in range(len(col_list)):
            x_list[i] = col_list[i]
        return x_list

# inside Autograd
col_list = [1, torch.tensor(2., requires_grad=True)]
x_list = Compress.apply(col_list)
print(x_list)

# outside
x_list = torch.ones((len(col_list), 1))
for i in range(len(col_list)):
    x_list[i] = col_list[i]

print(x_list)