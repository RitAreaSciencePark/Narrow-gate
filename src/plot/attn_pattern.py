
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from typing import Optional, List
import json


def plot_attn_pattern_plotly(
    tensor,
    string_tokens: Optional[List] = None,
    log_scale=False,
    normalize=False,
    vmin=None,
    vmax=None,
    width=300,
    height=300,
    save_path: Optional[str] = None,
    text_before_image=True,
    save_device = "html"
):
    """
    Plot attention pattern with options for better visualization using Plotly.

    Parameters:
    - tensor: square matrix of attention values
    - string_tokens: list of token strings
    - log_scale: if True, apply log scale to values
    - normalize: if True, normalize values to [0, 1] range
    - vmin, vmax: min and max values for color scaling
    - width, height: dimensions of the plot
    - save_path: path to save the HTML plot
    - text_before_image: if True, text tokens are at the beginning; if False, at the end
    """
    # compute the number of text tokens (image tokens 1024)
    n_text_tokens = tensor.size(0) - 1024 - 2
    print(f"Number of text tokens: {n_text_tokens}")

    # Convert tensor to numpy array
    data = tensor.cpu().numpy()

    # Determine text and image token ranges based on text_before_image
    if text_before_image:
        text_range = slice(0, n_text_tokens)
        image_range = slice(n_text_tokens, -2)
    else:
        text_range = slice(-n_text_tokens-2, -2)
        image_range = slice(0, -n_text_tokens-2)

    # compute the sum of the attention value (without the diagonal) of the two blocks
    text_text = data[text_range, text_range].sum() - np.trace(data[text_range, text_range])
    image_image = data[image_range, image_range].sum() - np.trace(data[image_range, image_range])
    image_text = data[image_range, text_range].sum() 

    # normalize by the number of tokens
    text_text /= n_text_tokens 
    image_image /= 1024 
    image_text /= (1024)
    print(f"Sum of attention values for text-text: {text_text}")
    print(f"Sum of attention values for image-image: {image_image}")

    if log_scale:
        data = np.log(data + 1e-9)  # Add small epsilon to avoid log(0)

    if normalize:
        data = (data - data.min()) / (data.max() - data.min())

    # Create hover text
    hover_text = [['' for _ in range(data.shape[1])] for _ in range(data.shape[0])]
    if string_tokens is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                hover_text[i][j] = f"From: \"{string_tokens[i]}\"<br>To: \"{string_tokens[j]}\""

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            colorscale=[[0, "white"], [0.5, "royalblue"], [1, "navy"]],
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Attention"),
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<br>%{text}<extra></extra>",
            text=hover_text
        )
    )

    # Add rectangle to highlight text token area
    # if text_before_image:
    #     x0, y0, x1, y1 = 0, 0, n_text_tokens, n_text_tokens
    # else:
    #     x0, y0 = tensor.size(0) - n_text_tokens - 2, tensor.size(0) - n_text_tokens - 2
    #     x1, y1 = tensor.size(0) - 2, tensor.size(0) - 2

    # fig.add_shape(
    #     type="rect",
    #     x0=x0, y0=y0, x1=x1, y1=y1,
    #     line=dict(color="Black", width=2),
    #     fillcolor="rgba(0,0,0,0)",
    # )

    # Update layout
    fig.update_layout(
        title="Attention Pattern",
        xaxis_title="Token Position",
        yaxis_title="Token Position",
        width=width,
        height=height,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
    )

    # # Add annotations for legend
    # fig.add_annotation(
    #     x=1.02, y=1, xref="paper", yref="paper",
    #     text="Text -> Image" if text_before_image else "Image -> Text",
    #     showarrow=False, font=dict(size=12), align="left",
    #     bgcolor="rgba(255,0,0,0.3)", bordercolor="black", borderwidth=1,
    # )
    # fig.add_annotation(
    #     x=1.02, y=0.9, xref="paper", yref="paper",
    #     text=f"Text-Text Attention: {text_text:.2f}",
    #     showarrow=False, font=dict(size=12), align="left",
    #     bgcolor="rgba(255,0,0,0.3)", bordercolor="black", borderwidth=1,
    # )
    # fig.add_annotation(
    #     x=1.02, y=0.85, xref="paper", yref="paper",
    #     text=f"Image-Image Attention: {image_image:.2f}",
    #     showarrow=False, font=dict(size=12), align="left",
    #     bgcolor="rgba(0,0,255,0.3)", bordercolor="black", borderwidth=1,
    # )
    # fig.add_annotation(
    #     x=1.02, y=0.8, xref="paper", yref="paper",
    #     text=f"Image-Text Attention: {image_text:.2f}",
    #     showarrow=False, font=dict(size=12), align="left",
    #     bgcolor="rgba(0,255,0,0.3)", bordercolor="black", borderwidth=1,
    # )

    if save_path:
        if save_device == "html":
            fig.write_html(f"{save_path}.html")
        if save_device == "json":
            json_fig= fig.to_json()
            json.dump(json_fig, open(f"{save_path}.json", "w"))
                
    else:
        fig.show()


def plot_attn_pattern(
    tensor,
    log_scale=False,
    normalize=False,
    vmin=None,
    vmax=None,
    figsize=(12, 10),
    save_path: Optional[str] = None,
):
    """
    Plot attention pattern with options for better visualization.

    Parameters:
    - tensor: square matrix of attention values
    - log_scale: if True, apply log scale to values
    - normalize: if True, normalize values to [0, 1] range
    - vmin, vmax: min and max values for color scaling
    - figsize: size of the figure
    """
    plt.figure(figsize=figsize)

    if log_scale:
        tensor = np.log(tensor + 1e-9)  # Add small epsilon to avoid log(0)

    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        # Create a custom colormap from white to deep blue
    colors = ["white", "royalblue", "navy"]
    n_bins = 100  # Number of color gradations
    cmap = LinearSegmentedColormap.from_list("white_to_blue", colors, N=n_bins)

    sns.heatmap(
        tensor, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": "Attention"}
    )
    plt.title("Attention Pattern")
    plt.xlabel("Token Position")
    plt.ylabel("Token Position")
    if save_path:
        plt.savefig(save_path)
    plt.show()