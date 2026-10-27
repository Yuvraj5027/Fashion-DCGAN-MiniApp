import pandas as pd
import streamlit as st
from torchvision.utils import make_grid
from backend import train, generate

st.set_page_config(page_title="Fashion DCGAN", layout="centered")
st.title("Fashion Image Generator (DCGAN)")


def to_grid(imgs):
    g = make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    return g.permute(1, 2, 0).numpy()


# ---------- Training controls ----------
st.sidebar.header("Training controls")
epochs = st.sidebar.slider("Epochs", 1, 50, 30)
subset = st.sidebar.slider("Dataset subset size", 1000, 10000, 3000, step=1000)
batch = st.sidebar.selectbox("Batch size", [64, 128, 256], index=1)
lr = st.sidebar.select_slider("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)

if st.sidebar.button("Train", type="primary"):
    status, bar, gallery = st.empty(), st.progress(0), st.empty()

    def on_epoch(ep, total, h):
        bar.progress(ep / total)
        status.write(f"**Epoch {ep}/{total}** | D loss {h['d_loss'][-1]:.3f} | G loss {h['g_loss'][-1]:.3f}")
        gallery.image(to_grid(h["samples"][-1]), caption=f"Generated images - epoch {ep}", width=500)

    G, D, history = train(epochs, subset, batch, lr, on_epoch)
    st.session_state["G"], st.session_state["history"] = G, history
    st.success("Training complete. Checkpoints saved in the 'checkpoints' folder.")

# ---------- Results ----------
if "G" in st.session_state:
    h = st.session_state["history"]
    st.subheader("Loss record")
    st.line_chart(pd.DataFrame({"Discriminator": h["d_loss"], "Generator": h["g_loss"]}))

    st.subheader("Progress gallery (first vs last epoch)")
    c1, c2 = st.columns(2)
    c1.image(to_grid(h["samples"][0]), caption="Epoch 1")
    c2.image(to_grid(h["samples"][-1]), caption=f"Epoch {len(h['samples'])}")

    st.subheader("New generated images")
    if st.button("Generate new images"):
        st.image(to_grid(generate(st.session_state["G"], 32)), width=500)
else:
    st.info("Set the controls in the sidebar and click Train.")
