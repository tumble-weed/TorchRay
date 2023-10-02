import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method',type=str,choices=['elp','elp_with_scale'],default='elp')
args = parser.parse_args()


if args.method == 'elp_with_scale':
    from torchray.attribution.extremal_perturbation_with_scale import extremal_perturbation, contrastive_reward,DELETE_VARIANT
elif args.method == 'elp':
    from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward,DELETE_VARIANT
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import get_device

# Obtain example data.
model, x, category_id_1, category_id_2 = get_example_data()

# Run on GPU if available.
device = get_device()
model.to(device)
x = x.to(device)


# Extremal perturbation backprop.
masks_1, _ = extremal_perturbation(
    model, x, category_id_1,
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.12],
    # variant = DELETE_VARIANT,
)


masks_2, _ = extremal_perturbation(
    model, x, category_id_2,
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.05],
    # variant = DELETE_VARIANT,
)



# Plots.
plot_example(x, masks_1, 'extremal perturbation', category_id_1)
plot_example(x, masks_2, 'extremal perturbation', category_id_2)
