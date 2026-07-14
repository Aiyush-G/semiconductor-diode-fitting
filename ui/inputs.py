"""
Reusable Streamlit input widgets for the diode-fitting pages.

Streamlit has no built-in "slider plus number box" control, and two widgets
cannot share a single ``session_state`` key. The helper here uses the standard
two-key + callback recipe: a slider and a number_input each own their own key,
and an ``on_change`` callback copies the edited value into the other so they
stay in sync. Users can drag the slider for quick exploration or type an exact
value into the box.
"""

import streamlit as st


def slider_with_number(
    label: str,
    *,
    min_value,
    max_value,
    value,
    step,
    key: str,
    help: str | None = None,
    fmt: str | None = None,
):
    """Render a slider and a number_input side-by-side that stay in sync.

    The number box is unbounded above ``max_value``: typing a larger value is
    allowed and the slider's upper bound expands to accommodate it. The expanded
    bound is sticky for the session (it grows but never shrinks) because
    shrinking a slider's ``max_value`` makes Streamlit reset the slider to its
    minimum. ``min_value`` remains a hard floor.

    Args:
        label: control label (shown on the slider; the number box mirrors it).
        min_value, max_value, value, step: base bounds/default/step. ``min_value``
            is a hard floor on both widgets; ``max_value`` is the initial slider
            top but the number box may exceed it (see above). Types must be
            consistent (all ``float`` or all ``int``).
        key: base session_state key; the widgets use ``{key}_sld`` / ``{key}_num``.
        help: tooltip shown on the slider label.
        fmt: optional printf-style format for the number box (e.g. "%.2f").

    Returns:
        The current value (kept identical across both widgets).
    """
    sld_key = f"{key}_sld"
    num_key = f"{key}_num"
    emax_key = f"{key}_emax"

    # Seed the keys once so neither widget needs a ``value=`` argument (passing
    # both ``value`` and an existing key triggers a Streamlit warning).
    if sld_key not in st.session_state:
        st.session_state[sld_key] = value
        st.session_state[num_key] = value
        st.session_state[emax_key] = max_value

    def _sync_from_slider() -> None:
        st.session_state[num_key] = st.session_state[sld_key]

    def _sync_from_number() -> None:
        st.session_state[sld_key] = st.session_state[num_key]

    # The number box is unbounded above so a user can type past the base range.
    # The slider's max grows to include the current value but is *sticky* — it
    # never shrinks, because shrinking a slider's max_value makes Streamlit reset
    # the slider to its minimum (which caused a snap-to-zero while dragging down).
    current = st.session_state[num_key]
    slider_max = max(st.session_state[emax_key], current)
    st.session_state[emax_key] = slider_max

    col_slider, col_number = st.columns([3, 1], gap="small")
    with col_slider:
        st.slider(
            label,
            min_value=min_value,
            max_value=slider_max,
            step=step,
            key=sld_key,
            help=help,
            on_change=_sync_from_slider,
        )
    with col_number:
        number_kwargs = dict(
            min_value=min_value,
            step=step,
            key=num_key,
            on_change=_sync_from_number,
            label_visibility="collapsed",
        )
        if fmt is not None:
            number_kwargs["format"] = fmt
        st.number_input(label, **number_kwargs)

    return st.session_state[sld_key]
