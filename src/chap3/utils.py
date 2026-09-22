"""Plot audio at full scale, inspect its samples, or animate a gradual zoom.

The plotting functions return a Matplotlib figure for notebook display or saving.
``save_waveform_zoom_gif`` writes an animation and returns its path.
"""

from pathlib import Path
from numbers import Integral

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf

__all__ = ["plot_waveform", "plot_waveform_zoom", "save_waveform_zoom_gif"]

_LINE_COLOR = '#2563eb'
_SAMPLE_COLOR = '#f97316'


def _load_audio(audio_path, start=0.0, duration=None, channel=0):
    """Read one channel; a missing duration reads to the end of the file."""
    if not np.isfinite(start) or start < 0:
        raise ValueError('start must be finite and nonnegative')
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError('duration must be finite and positive')

    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = audio_file.samplerate
        if not isinstance(channel, Integral) or not 0 <= channel < audio_file.channels:
            raise ValueError(f'channel must be between 0 and {audio_file.channels - 1}')
        first_sample = round(start * sample_rate)
        if first_sample >= len(audio_file):
            raise ValueError('start is past the end of the audio')
        audio_file.seek(first_sample)
        frames = -1 if duration is None else max(1, round(duration * sample_rate))
        data = audio_file.read(frames=frames, dtype='float64', always_2d=True)

    channel_data = data[:, channel]
    if len(channel_data) < 2:
        raise ValueError('snippet needs at least two samples')
    if not np.isfinite(channel_data).all():
        raise ValueError('audio contains non-finite sample values')

    return channel_data, sample_rate


def _validate_samples(samples):
    if not isinstance(samples, Integral) or samples < 2:
        raise ValueError('samples must be an integer of at least 2')


def _zoom_geometry(sample_count, sample_rate, samples):
    """Return a centered sample window and its initial/final widths in ms."""
    count = min(samples, sample_count)
    left = (sample_count - count) // 2
    center = (left + (count - 1) / 2) / sample_rate * 1000
    final_width = (count + 1) / sample_rate * 1000
    # Include both ends even when the selected sample window is off-center.
    last_time = (sample_count - 1) / sample_rate * 1000
    initial_width = max(2 * center, 2 * (last_time - center), final_width)
    return left, count, center, initial_width, final_width


def _style_axis(ax, data, xlabel):
    limit = max(float(np.max(np.abs(data))) * 1.35, 0.05)
    ax.set(xlabel=xlabel, ylabel='Amplitude', ylim=(-limit, limit))
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='#94a3b8', lw=0.8, zorder=0)
    ax.grid(alpha=0.16)


def plot_waveform(audio_path: str | Path, *, channel: int = 0) -> Figure:
    """Display the entire audio channel, with time in seconds.

    Returns a figure without saving it or calling ``plt.show()``.
    """
    data, sample_rate = _load_audio(audio_path, channel=channel)
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    ax.plot(np.arange(len(data)) / sample_rate, data, color=_LINE_COLOR, lw=0.8)
    _style_axis(ax, data, 'Time (s)')
    ax.set_xlim(0, (len(data) - 1) / sample_rate)
    ax.set_title(f'{Path(audio_path).name} | {sample_rate:,} samples/sec | channel {channel}')
    return fig


def plot_waveform_zoom(
    audio_path: str | Path, *, start: float = 0.0, duration: float = 0.1,
    channel: int = 0, samples: int = 24, stages: int = 4,
) -> Figure:
    """Show successive zoom levels as stacked panels, ending at sample dots.

    ``start`` and ``duration`` are seconds. Each panel zooms into the same
    centered region; ``samples`` controls the final number of visible samples.
    Time is relative to the snippet start. Returns an unsaved figure.
    """
    _validate_samples(samples)
    if not isinstance(stages, Integral) or stages < 2:
        raise ValueError('stages must be an integer of at least 2')
    data, sample_rate = _load_audio(audio_path, start, duration, channel)
    _, _, center, initial_width, final_width = _zoom_geometry(len(data), sample_rate, samples)
    time_ms = np.arange(len(data)) / sample_rate * 1000
    fig, axes = plt.subplots(stages, 1, figsize=(11, 1.5 * stages), layout='constrained')
    fig.suptitle(f'{Path(audio_path).name}: from waveform to samples', fontweight='bold')
    for index, (ax, width) in enumerate(zip(axes, np.geomspace(initial_width, final_width, stages))):
        ax.plot(time_ms, data, color=_LINE_COLOR, lw=1.2)
        if width * sample_rate / 1000 <= 90 or index == stages - 1:
            ax.plot(time_ms, data, linestyle='none', marker='o', color=_SAMPLE_COLOR, markersize=5)
        # _style_axis(ax, data, 'Time from snippet start (ms)')
        ax.set_xlim(center - width / 2, center + width / 2)
        ax.set_title(f'{initial_width / width:.1f}x zoom | {width:.2f} ms visible', loc='left')
    return fig


def _create_zoom_animation(data, sample_rate, audio_name, channel, samples):
    """Build the animation figure and its frame-update callback."""
    left, count, center, initial_width, final_width = _zoom_geometry(len(data), sample_rate, samples)
    time_ms = np.arange(len(data)) / sample_rate * 1000

    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(top=0.76, bottom=0.25, left=0.10, right=0.96)
    fig.suptitle('Look closer: a waveform is made of samples', fontsize=21, fontweight='bold', y=0.96)
    fig.text(0.5, 0.885, f'{audio_name}  |  {sample_rate:,} samples/sec  |  channel {channel}', ha='center', color='#475569')

    stage = fig.text(0.5, 0.815, '', ha='center', fontsize=14, color=_LINE_COLOR)
    line, = ax.plot(time_ms, data, color=_LINE_COLOR, lw=1.6)
    dots, = ax.plot(time_ms, data, linestyle='none', marker='o', color=_SAMPLE_COLOR, markersize=6, alpha=0, zorder=4)
    _style_axis(ax, data, 'Time from snippet start (ms)')

    labels = []
    for sample_index in np.linspace(left, left + count - 1, min(3, count), dtype=int):
        label = ax.annotate(
            f'x[{sample_index}] = {data[sample_index]:+.3f}',
            (time_ms[sample_index], data[sample_index]),
            xytext=(0, 17 if data[sample_index] >= 0 else -25),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            color='#9a3412',
            alpha=0
        )
        labels.append(label)

    status = fig.text(0.5, 0.135, '', ha='center', fontsize=12)
    # fig.text(
    #     0.5, 0.055,
    #     'The line connects decoded samples for display. Each dot is one amplitude.\n'
    #     'Sample spacing = 1 / sample rate. Animation is slowed down and silent.',
    #     ha='center', fontsize=10, color='#475569'
    # )

    def update(progress):
        width = initial_width * (final_width / initial_width) ** progress
        ax.set_xlim(center - width / 2, center + width / 2)
        visible = width * sample_rate / 1000
        reveal = float(np.clip((90 - visible) / 60, 0, 1))
        dots.set_alpha(reveal)
        line.set_alpha(1 - 0.65 * reveal)
        for label in labels:
            label.set_alpha(float(np.clip((progress - 0.88) / 0.12, 0, 1)))
        stage.set_text(
            'A waveform at normal scale' if progress < 0.1 else
            'Zooming into the same audio...' if progress < 0.95 else
            'Individual samples become visible'
        )
        status.set_text(
            f'{initial_width / width:.1f}x zoom   |   '
            f'{width:.2f} ms visible   |   '
            f'one sample every {1000 / sample_rate:.4f} ms'
        )
        return line, dots, stage, status, *labels

    return fig, update


def _save_animation(fig, update_frame, output_path, dpi):
    """Hold the overview, zoom in smoothly, hold the samples, then zoom out."""
    progress = np.linspace(0, 1, 72)
    eased = progress * progress * (3 - 2 * progress)
    frames = np.r_[np.zeros(16), eased, np.ones(32), eased[::-1], np.zeros(8)]
    animation = FuncAnimation(fig, update_frame, frames=frames, interval=50, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=20), dpi=dpi)


def save_waveform_zoom_gif(
    audio_path: str | Path,
    output_path: str | Path = Path('audio_zoom_output/waveform_zoom.gif'),
    *, start: float = 0.0, duration: float = 0.1, channel: int = 0,
    samples: int = 24, dpi: int = 200,
) -> Path:
    """Save the silent, looping zoom animation and return the GIF path.

    ``start`` and ``duration`` are seconds; ``samples`` controls the final
    sample window. The figure is closed after saving. Parent folders are
    created automatically. Lower ``dpi`` reduces output size and render cost.
    """
    _validate_samples(samples)
    if not isinstance(dpi, Integral) or dpi <= 0:
        raise ValueError('dpi must be a positive integer')
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.gif':
        raise ValueError('output_path must end in .gif')
    data, sample_rate = _load_audio(audio_path, start, duration, channel)
    fig, update_frame = _create_zoom_animation(data, sample_rate, Path(audio_path).name, channel, samples)
    try:
        _save_animation(fig, update_frame, output_path, dpi)
    finally:
        plt.close(fig)
    return output_path


def generate_audio_zoom_visualization(
    audio_path: str,
    start=0.0,
    duration=0.1,
    channel=0,
    samples=24,
    output_dir=Path('audio_zoom_output'),
    save_gif=False
):
    """Compatibility wrapper for existing notebooks; prefer the three APIs above.

    Saves the final zoom to ``waveform.png`` and optionally the GIF, returning
    the final zoom figure as before.
    """
    _validate_samples(samples)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(audio_path)

    data, sample_rate = _load_audio(audio_path, start, duration, channel)
    fig, update_frame = _create_zoom_animation(data, sample_rate, audio_path.name, channel, samples)
    update_frame(1.0)
    fig.savefig(output_dir / 'waveform.png', dpi=400)
    if save_gif:
        _save_animation(fig, update_frame, output_dir / 'waveform_zoom.gif', dpi=200)
        update_frame(1.0)
    return fig
