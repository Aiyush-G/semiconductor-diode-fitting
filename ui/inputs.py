"""
Reusable Streamlit input widgets for the diode-fitting pages.

Streamlit has no built-in "slider plus number box" control, and two widgets
cannot share a single ``session_state`` key. The helper here uses the standard
two-key + callback recipe: a slider and a number_input each own their own key,
and an ``on_change`` callback copies the edited value into the other so they
stay in sync. Users can drag the slider for quick exploration or type an exact
value into the box.
"""

import math

import streamlit as st


def saturation_current_input(
    label: str = "Saturation current density J_0 (A/cm²)",
    *,
    key: str = "j_0",
    default: float = 1e-13,
    decade_range: tuple[int, int] = (-15, -5),
    min_value: float = 1e-16,
    max_value: float = 1e-5,
    help: str | None = None,
    fit_key: str | None = None,
    fit_disabled: bool = False,
) -> float:
    """Decade select-slider + synced number box for a saturation current J_0.

    J_0 spans many orders of magnitude, so the select-slider gives a quick
    decade sweep while the number box allows an exact value (e.g. 3e-14). The
    number box is the source of truth for the model; when it changes, the decade
    selector snaps to the nearest option so the two stay visually consistent.

    When ``fit_key`` is given, a "Fit" checkbox is rendered as a third column
    (matching ``slider_with_number``); the caller reads ``st.session_state[fit_key]``.

    Args:
        label: control label (shown on the select-slider; the number box mirrors it).
        key: base session_state key; the widgets use ``{key}_sel`` / ``{key}_num``.
        default: initial J_0 value (A/cm²), seeded once into session state.
        decade_range: (lowest, highest) power of ten offered by the decade
            selector; ``range(lo, hi)`` exclusive of ``hi``, matching Python.
        min_value, max_value: hard bounds on the free-form number box (span the
            decade selector with a little headroom on either side).
        help: tooltip shown on the select-slider label.
        fit_key: if given, render a "Fit" checkbox with this session_state key.
        fit_disabled: render the Fit checkbox disabled (e.g. on dark data).

    Returns:
        The current J_0 value (A/cm²).
    """
    options = [10 ** p for p in range(decade_range[0], decade_range[1])]
    sel_key = f"{key}_sel"
    num_key = f"{key}_num"

    if num_key not in st.session_state:
        st.session_state[num_key] = default
        st.session_state[sel_key] = min(
            options, key=lambda option: abs(math.log10(option) - math.log10(default))
        )

    def _sync_from_select() -> None:
        st.session_state[num_key] = st.session_state[sel_key]

    def _sync_from_number() -> None:
        value = st.session_state[num_key]
        # Snap the decade selector to the option closest in log10 space.
        st.session_state[sel_key] = min(
            options,
            key=lambda option: abs(math.log10(option) - math.log10(value)),
        )

    if help is None:
        help = (
            "Reverse saturation current density in A/cm². This controls the diode "
            "recombination current and strongly affects open-circuit voltage."
        )
    if fit_key is not None:
        col_select, col_number, col_fit = st.columns([3, 1, 1], gap="small")
    else:
        col_select, col_number = st.columns([3, 1], gap="small")
    with col_select:
        st.select_slider(
            label,
            options=options,
            format_func=lambda x: f"{x:.0e}",
            key=sel_key,
            help=help,
            on_change=_sync_from_select,
        )
    with col_number:
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            step=abs(st.session_state[num_key]) / 10 or 1e-14,
            format="%.2e",
            key=num_key,
            on_change=_sync_from_number,
            label_visibility="collapsed",
        )
    if fit_key is not None:
        if fit_key not in st.session_state:
            st.session_state[fit_key] = True
        with col_fit:
            st.checkbox("Fit", key=fit_key, disabled=fit_disabled)

    return st.session_state[num_key]


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
    fit_key: str | None = None,
    fit_help: str | None = None,
    fit_disabled: bool = False,
):
    """Render a slider and a number_input side-by-side that stay in sync.

    The number box is unbounded above ``max_value``: typing a larger value is
    allowed and the slider's upper bound expands to accommodate it. The expanded
    bound is sticky for the session (it grows but never shrinks) because
    shrinking a slider's ``max_value`` makes Streamlit reset the slider to its
    minimum. ``min_value`` remains a hard floor.

    Optionally, a "Fit" checkbox can be rendered as a third column (used by the
    fitting workflow to mark this parameter free/fixed). It is added *inside* this
    control's own column row rather than as an outer wrapper, so it does not add
    an extra level of column nesting.

    Args:
        label: control label (shown on the slider; the number box mirrors it).
        min_value, max_value, value, step: base bounds/default/step. ``min_value``
            is a hard floor on both widgets; ``max_value`` is the initial slider
            top but the number box may exceed it (see above). Types must be
            consistent (all ``float`` or all ``int``).
        key: base session_state key; the widgets use ``{key}_sld`` / ``{key}_num``.
        help: tooltip shown on the slider label.
        fmt: optional printf-style format for the number box (e.g. "%.2f").
        fit_key: if given, render a "Fit" checkbox with this session_state key in a
            third column. The caller reads the state from ``st.session_state[fit_key]``.
        fit_help: tooltip for the Fit checkbox.
        fit_disabled: render the Fit checkbox disabled (e.g. J_ph on dark data).

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

    if fit_key is not None:
        col_slider, col_number, col_fit = st.columns([3, 1, 1], gap="small")
    else:
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
    if fit_key is not None:
        if fit_key not in st.session_state:
            st.session_state[fit_key] = True
        with col_fit:
            st.checkbox("Fit", key=fit_key, help=fit_help, disabled=fit_disabled)

    return st.session_state[sld_key]
