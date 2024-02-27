import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import gradio as gr

from phonax.gradio import (
    predict_ph_v1,
)

# phonax - gradio interface : v1

demo = gr.Interface(
    fn=predict_ph_v1,
    inputs=[gr.Checkbox(label="Use finetuned model", info="PBEsol finetuned phonon model"),
        gr.Checkbox(label="Find primitive unit cell", info="Convert the lattice into primitive unit cell"),
        gr.Checkbox(label="\u0393-point only", info="Only compute the DOS at q=0"),
        "file",
    ],
    outputs="image"
)

# set share for public sharing link
demo.launch(server_name = '0.0.0.0',server_port = 7860, share = False)

