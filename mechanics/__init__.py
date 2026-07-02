"""
mechanics/
==========
The Mechanics tab — D2IM, a deep-learning model that predicts the *mechanics*
of a bone from its image alone.

Where the Vision tab answers "what am I looking at?" qualitatively, the
Mechanics tab answers "how does it deform?" quantitatively: given a single
undeformed micro-CT (XCT) slice, the D2IM CNN predicts the displacement field
(u, v, w) and, by spatial differentiation of the axial component, the axial
strain field ε_zz. This is the group's own model (Soar, Palanca, Dall'Ara &
Tozzi, J. Orthop. Translat. 2024) wired in as a fully isolated, swappable
adapter — everything downstream depends on the prediction contract in
``d2im.predict``, never on TensorFlow specifics.

d2im    The D2IM inference adapter: lazy-loads the Keras model, replicates the
        paper's pre-/post-processing, and renders displacement + strain fields.
        Importing this module never pulls in TensorFlow; that happens only when
        a prediction is actually requested.

> Research and educational use only — not a clinical tool.
"""
