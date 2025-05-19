import pandas as pd
import ast
import numpy as np
from typing import Optional, List


def split_line(line):
    fields = []
    field = ""
    in_single_quote = False
    in_double_quote = False
    in_brackets = 0
    i = 0
    while i < len(line):
        c = line[i]
        if c == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            field += c
        elif c == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            field += c
        elif c == "[" and not in_single_quote and not in_double_quote:
            in_brackets += 1
            field += c
        elif c == "]" and not in_single_quote and not in_double_quote:
            in_brackets -= 1
            field += c
        elif (
            c == ","
            and not in_single_quote
            and not in_double_quote
            and in_brackets == 0
        ):
            fields.append(field.strip())
            field = ""
        else:
            field += c
        i += 1
    if field:
        fields.append(field.strip())
    return fields


def strip_quotes(s):
    s = s.strip()
    # Remove leading and trailing single or double quotes
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]
    return s


def parse_list(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        # Use ast.literal_eval to safely evaluate the list
        return ast.literal_eval(s)
    else:
        return s


def clean_activation_path(s):
    s = s.strip()
    # Remove any trailing parentheses
    s = s.rstrip(")")
    # Remove leading and trailing quotes
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]
    # Replace escaped single quotes with actual single quotes
    s = s.replace("\\'", "'")
    return s


def load_activations_df(activation_txt):
    # Read the data from the text file
    with open(activation_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_line = lines[0]
    column_names = [name.strip() for name in header_line.strip().split(",")]

    # Initialize a list to hold dictionaries for each line
    data_list = []

    # Loop through the remaining lines
    for line in lines[1:]:
        # Extract data values
        data_values = split_line(line.strip())

        # Create a dictionary for the current line
        data_dict = dict(zip(column_names, data_values))

        # Clean and parse each field appropriately
        data_dict["patching_elem"] = strip_quotes(data_dict["patching_elem"])
        data_dict["layers_to_patch"] = parse_list(data_dict["layers_to_patch"])
        data_dict["activation_type"] = strip_quotes(data_dict["activation_type"])
        data_dict["activation_path"] = clean_activation_path(
            data_dict["activation_path"]
        )

        # Add the dictionary to the list
        data_list.append(data_dict)

    # Create the pandas DataFrame from the list of dictionaries
    activations_path = pd.DataFrame(data_list)
    return activations_path


import matplotlib.pyplot as plt
import torch


def pre_process_data(df, key):
    """
    Plots the mean of the specified activation key against the layers_to_patch (as strings).

    Parameters:
    - df: pandas DataFrame containing 'activation_path' and 'layers_to_patch'
    - key: The key in data['activations'] to compute the mean from
    """
    # Dictionaries to store mean values for each layers_to_patch
    patched_data_dict = {}
    clean_data_dict = {}
    dist_diff = {}
    dist_diff_std = {}
    patched_data_dict_std = {}

    for index, row in df.iterrows():
        activation_path = row["activation_path"]
        layers_to_patch = row["layers_to_patch"]

        # Convert layers_to_patch to a string for categorical x-axis
        layers_str = str(layers_to_patch[0])

        # Load the activation data
        data = torch.load(activation_path, map_location="cpu")
        print(layers_str)
        # Get the mean activation value for the given key
        if key == "logit_diff":
            logit_diff_patched = data["logit_diff_in_patched"]
            logit_diff_clean = data["logit_diff_in_clean"]

            # count the number of sign differences between the two tensors
            sign_diff = torch.sum(
                torch.sign(data
                ["logit_diff_in_patched"])
                != torch.sign(data
                ["logit_diff_in_clean"]),
                -1,
            )
            dist_diff[layers_str] = sign_diff.item() / logit_diff_patched.shape[0]
            print(dist_diff[layers_str])

            

        elif key == "dist_similarity":
            cat_meno_dog_patched = torch.softmax(
                data["target_patched_logits"], -1
            )
            cat_meno_dog_clean = torch.softmax(
                data["target_clean_logits"], -1
            )
            cat_clean = torch.softmax(data["base_logits"], -1)

            # subtract the two tensors

            dist_sum = torch.sum(torch.min(cat_meno_dog_patched, cat_clean), -1)

            base = torch.sum(torch.min(cat_meno_dog_clean, cat_clean), -1).mean().item()

            dist_diff[layers_str] = dist_sum.mean().item()
            dist_diff_std[layers_str] = dist_sum.std().item()
            print(dist_diff[layers_str])

        else:
            tensor = data["activations"][key]
            if tensor.is_cuda:
                tensor = tensor.cpu()
            mean_value = tensor.mean().item()
            patched_data_dict[layers_str] = mean_value
            clean_data_dict[layers_str] = mean_value

    # print(dist_diff)
    # Prepare data for plotting
    x_values = list(dist_diff.keys())
    dist_y_values = [dist_diff[key] for key in x_values]


    # Create the bar plot
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    x_indices = range(len(x_values))

    # if key == "dist_similarity":
    print(dist_y_values)
    plt.bar(
        x_indices,
        dist_y_values,
        width=bar_width,
        label="Dist Similarity",
        color="r",
        alpha=1,
    )
    # plt.errorbar(
    #     x_indices,
    #     dist_y_values,
    #     yerr=[dist_diff_std[key] for key in x_values],
    #     fmt="o",
    #     color="k",
    #     label="Std Dev",
    # )
    # else:
    #     # Plotting both patched and clean logit differences
    #     plt.bar(
    #         x_indices,
    #         patched_y_values,
    #         width=bar_width,
    #         label="Patched",
    #         color="b",
    #         alpha=1,
    #     )
    #     plt.bar(
    #         [x + bar_width for x in x_indices],
    #         clean_y_values,
    #         width=bar_width,
    #         label="Clean",
    #         color="g",
    #         alpha=0.6,
    #     )

    # Adding labels and title
    plt.xlabel("Layers Patched")
    plt.ylabel(f"Logit(Cat) - Logit(Dog)")
    plt.title(f"Comparison of Logit Differences between Patched and Clean Conditions")
    plt.xticks(
        [x + bar_width / 2 for x in x_indices], x_values, rotation=45, ha="right"
    )

    # Adding legend
    plt.legend()
    plt.tight_layout()
    plt.show()
    if key == "dist_similarity":
        return x_indices, dist_y_values, [dist_diff_std[key] for key in x_values], base
    if key == "logit_diff":
        return x_indices, dist_y_values, None, None


def barplot_dist_similarity(
    x_indices,
    dist_y_values,
    dist_y_err,
    base,
    axvspan_low,
    axvspan_high,
    save_path,
    save=True,
):
    
    
    large_font = 16
    medium_font = 15
    small_font = 13
    # palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # palette2 = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB']
    # palette_light = [ '#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
    # muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]
    bright = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]


    
    # x_indices = [f"{i}" for i in range(len(dist_y_values))]
    # x_indices = ["[0,1,2,3]", "[4,5,6,7]", "[8,9,10,11]", "[12,13,14,15]", "[16,17,18,19]", "[20,21,22,23]", "[24,25,26,27]", "[28,29,30,31]"]
    # Define the high-contrast color palette
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]

    # Refine the barplot
    # plt.figure(figsize=(32,6))
    plt.figure(figsize=(10.0, 6.15))
    bar_width = 0.8
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.axvspan(axvspan_low, axvspan_high, color=bright[3], alpha=0.25, lw=0)
    bars = plt.bar(
        x_indices,
        dist_y_values,
        width=bar_width,
        label="Target Distribution After Patching",
        color=bright[1],
        alpha=0.99,
    )
    for bar in bars:
        bar.set_linewidth(1.5)
        bar.set_edgecolor("black")
        bar.set_linestyle("-")
        bar.set_capstyle("round")
        # bar.set_edgealpha(0.9)
        # add alpha to the edge color
        # bar.set_alpha(0.4)
    plt.errorbar(
        x_indices,
        dist_y_values,
        yerr=dist_y_err,
        fmt="o",
        color="black",
        alpha=0.99,
        linewidth=1.6,
        markersize=0,
    )

    # add the base line
    plt.axhline(
        y=base,
        color=hg_contrast[0],
        linestyle="-",
        label="Target Distribution Without Patching",
        linewidth=3,
    )

    # axvspan

    plt.xlabel("Layer where activations were patched", fontsize=medium_font)
    plt.ylabel("Similarity to base distribution", fontsize=medium_font)
    plt.title(
        "Layer-wise Similarity Shift After Patching Residual Stream at <end-image>",
        fontsize=large_font,
    )
    # plot just some of the x-ticks 0, 4 8, 12, 16, 20, 24, 28, 31
    selected_x_indices = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    plt.xticks(
        selected_x_indices,    selected_x_indices = [0, 4, 8, 12, 16, 20, 24, 28, 31]
, rotation=0, ha="center", fontsize=small_font
    )
    
    # plt.xticks(
    #     x_indices, [f"{i}" for i in x_indices], rotation=0, ha="center", fontsize=small_font
    # )

    # Add a grid and specify more labels for y-axis
    y_ticks = np.linspace(0, 1, 5)
    y_tick_labels = (
        ["Less\nSimilar"]
        + [f"{round(tick, 2)}" for tick in y_ticks[1:-1]]
        + ["Most\nSimilar"]
    )
    plt.yticks(y_ticks, y_tick_labels, fontsize=small_font)
    plt.legend(fontsize=small_font, loc="upper left")

    plt.tight_layout()
    # plt.show()
    if save == True:
        plt.savefig(save_path, format="pdf", dpi=300)
    else:
        return plt


def lineplot_dist_similarity(
    x_indices,
    dist_y_values,
    dist_y_err: Optional[List],
    base: Optional[float],
    axvspan_low,
    axvspan_high,
    title,
    save_path,
    save=True,
):
        
    large_font = 26
    medium_font = 24
    small_font = 22
    # Define the high-contrast and bright color palettes
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]
    bright = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]

    # Create the line plot
    plt.figure(figsize=(10.0, 6.15))
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    # plt.axvspan(axvspan_low, axvspan_high, color=bright[3], alpha=0.3, lw=0)

    if base is not None:
    # Add the base line
        plt.axhline(
            y=base,
            color=hg_contrast[0],
            linestyle="--",
            label="Without Patching",
            linewidth=4,
        )

    # Plot the line
    plt.plot(
        x_indices,
        dist_y_values,
        "-o",
        color=bright[2],
        alpha=0.99,
        linewidth=4,
        markersize=9,
        label="After Patching",
    )
    if dist_y_err is not None:
        # Plot shaded area for error instead of error bars
        lower_bound = np.array(dist_y_values) - np.array(dist_y_err)
        upper_bound = np.array(dist_y_values) + np.array(dist_y_err)
        plt.fill_between(x_indices, lower_bound, upper_bound, color=bright[2], alpha=0.3)

    # Labels and title
    plt.xlabel("Start indices of patched activations", fontsize=medium_font)
    # plt.ylabel("Similarity to base distribution\n", fontsize=medium_font)
    plt.title(
        title,
        fontsize=large_font,
    )


    # Y-axis customization
    y_ticks = np.linspace(0, 1, 5)
    y_tick_labels = (
        ["Less\nSimilar"]
        + [f"{round(tick, 2)}" for tick in y_ticks[1:-1]]
        + ["Most\nSimilar"]
    )
    plt.yticks(y_ticks, y_tick_labels, fontsize=small_font)
    plt.legend(fontsize=small_font-4.6, loc="upper right" , ncol=1)
    selected_x_indices = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    selected_x_labels = [f"{i}" for i in selected_x_indices]
    plt.xticks(
        selected_x_indices,    selected_x_labels, rotation=0, ha="center", fontsize=small_font
        )
    plt.tight_layout()

    # Save or return the plot
    if save:
        plt.savefig(save_path, format="pdf", dpi=300)
    else:
        return plt


def barplot_logit_diff(
    x_indices, dist_y_values, dist_y_err, base, axvspan_low, axvspan_high, save_path
):
    # palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # palette2 = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB']
    # palette_light = [ '#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
    # muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]
    bright = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]

    # x_indices = [f"{i}" for i in range(len(dist_y_values))]
    # x_indices = ["[0,1,2,3]", "[4,5,6,7]", "[8,9,10,11]", "[12,13,14,15]", "[16,17,18,19]", "[20,21,22,23]", "[24,25,26,27]", "[28,29,30,31]"]
    # Define the high-contrast color palette
    hg_contrast = ["#004488", "#DDAA33", "#BB5566"]

    # Refine the barplot
    plt.figure(figsize=(32, 6))
    bar_width = 0.7
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.axvspan(axvspan_low, axvspan_high, color=bright[3], alpha=0.25, lw=0)
    bars = plt.bar(
        x_indices,
        dist_y_values,
        width=bar_width,
        label="Target Distribution After Patching",
        color=bright[1],
        alpha=0.99,
    )
    for bar in bars:
        bar.set_linewidth(3.4)
        bar.set_edgecolor("black")
        bar.set_linestyle("-")
        bar.set_capstyle("round")
        # bar.set_edgealpha(0.9)
        # add alpha to the edge color
        # bar.set_alpha(0.4)
    plt.errorbar(
        x_indices,
        dist_y_values,
        yerr=dist_y_err,
        fmt="o",
        color="black",
        alpha=0.99,
        linewidth=3.4,
        markersize=10,
    )

    # add the base line
    plt.axhline(
        y=base,
        color=hg_contrast[0],
        linestyle="-",
        label="Target Distribution Without Patching",
        linewidth=8,
    )

    # axvspan

    plt.xlabel("Layer where activations were patched", fontsize=36)
    plt.ylabel("Similarity to\nbase distribution\n", fontsize=36)
    plt.title(
        "Layer-wise Similarity Shift After Patching Residual Stream at <end-image>",
        fontsize=38,
    )
    plt.xticks(
        x_indices, [f"{i}" for i in x_indices], rotation=0, ha="center", fontsize=32
    )

    # Add a grid and specify more labels for y-axis
    y_ticks = np.linspace(0, 1, 5)
    y_tick_labels = (
        ["Less\nSimilar"]
        + [f"{round(tick, 2)}" for tick in y_ticks[1:-1]]
        + ["Most\nSimilar"]
    )
    plt.yticks(y_ticks, y_tick_labels, fontsize=34)
    plt.legend(fontsize=34, loc="upper left")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, format="pdf", dpi=300)
