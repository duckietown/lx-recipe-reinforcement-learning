"""Training loops, evaluation and plotting used by the notebooks.

The learning algorithms themselves (Q-learning update, DQN network, replay buffer, TD loss) are
written by the learner in the ``solution`` package; the functions here only drive them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------ tabular RL


def run_tabular_episode(
    env,
    q_table: np.ndarray,
    choose_action: Callable[[np.ndarray, int, float, np.random.Generator], int],
    update: Callable[[np.ndarray, int, int, float, int, bool, float, float], None] | None,
    *,
    epsilon: float,
    alpha: float,
    gamma: float,
    rng: np.random.Generator,
    max_steps: int = 100,
) -> tuple[float, int, str]:
    """Run one episode on a tabular environment; ``update=None`` evaluates without learning.

    Returns:
        ``(return, length, event)`` where ``event`` is ``"duckie"``, ``"grass"`` or ``"timeout"``.

    """
    state, _ = env.reset()
    total, event = 0.0, "timeout"
    for t in range(max_steps):
        action = choose_action(q_table, state, epsilon, rng)
        next_state, reward, terminated, truncated, info = env.step(action)
        if update is not None:
            update(q_table, state, action, reward, next_state, terminated, alpha, gamma)
        total += reward
        state = next_state
        if terminated or truncated:
            event = info.get("event", "timeout")
            break
    return total, t + 1, event


def train_tabular(  # noqa: PLR0913
    env,
    choose_action,
    update,
    *,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    epsilon_final: float | None = None,
    seed: int = 0,
    max_steps: int = 100,
    q_init: float = 0.0,
) -> tuple[np.ndarray, dict[str, list]]:
    """Train a Q-table with the learner's ``choose_action`` and ``update`` functions.

    ``epsilon`` decays linearly to ``epsilon_final`` over the run when the latter is given. The table
    starts filled with ``q_init``.

    Returns:
        the Q-table and a history dict with per-episode ``returns``, ``lengths`` and ``events``.

    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    q_table = np.full((env.n_states, env.n_actions), float(q_init))
    history: dict[str, list] = {"returns": [], "lengths": [], "events": []}
    for episode in range(episodes):
        eps = epsilon
        if epsilon_final is not None and episodes > 1:
            eps = epsilon + (epsilon_final - epsilon) * episode / (episodes - 1)
        ret, length, event = run_tabular_episode(
            env, q_table, choose_action, update, epsilon=eps, alpha=alpha, gamma=gamma, rng=rng,
            max_steps=max_steps,
        )
        history["returns"].append(ret)
        history["lengths"].append(length)
        history["events"].append(event)
    return q_table, history


def moving_average(values, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ------------------------------------------------------------------------------ deep RL (DQN)


@dataclass
class DQNConfig:
    """Hyper-parameters of the DQN training loop (the defaults train in about 45 min on a laptop CPU)."""

    total_steps: int = 15_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.95
    learning_rate: float = 2.5e-4
    train_every: int = 2
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 6_000
    buffer_size: int = 30_000
    checkpoint_every: int = 5_000
    seed: int = 0

    def epsilon(self, step: int) -> float:
        frac = min(step / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)


@dataclass
class DQNHistory:
    """What the training loop records (saved as JSON next to the checkpoints)."""

    steps: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_end_steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    loss_steps: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    wall_time: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> DQNHistory:
        return cls(**json.loads(Path(path).read_text()))


def train_dqn(  # noqa: PLR0913, PLR0912, PLR0915
    env,
    q_net,
    target_net,
    buffer,
    *,
    select_action: Callable,
    compute_loss: Callable,
    config: DQNConfig,
    checkpoint_dir: str | Path,
    on_episode_end: Callable[[DQNHistory, int], None] | None = None,
    resume: bool = True,
) -> DQNHistory:
    """The DQN training loop.

    Args:
        env: environment from :func:`rl_duckiematrix.env.make_env` (discrete actions).
        q_net, target_net: the learner's ``QNetwork`` instances (``torch.nn.Module``).
        buffer: the learner's ``ReplayBuffer`` with ``add(obs, action, reward, next_obs, done)``,
            ``sample(batch_size) -> batch`` and ``__len__``.
        select_action: ``select_action(q_net, observation, epsilon) -> int``.
        compute_loss: ``compute_loss(q_net, target_net, batch, gamma) -> torch scalar``.
        config: :class:`DQNConfig`.
        checkpoint_dir: where ``q_net.pt``, ``optimizer.pt`` and ``history.json`` are written every
            ``config.checkpoint_every`` steps.
        on_episode_end: callback ``(history, step)`` for live plots.
        resume: continue from the checkpoint in ``checkpoint_dir`` when there is one.

    """
    import torch  # noqa: PLC0415

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    history = DQNHistory()
    step = 0
    if resume and (checkpoint_dir / "q_net.pt").exists():
        q_net.load_state_dict(torch.load(checkpoint_dir / "q_net.pt", map_location="cpu"))
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu"))
        history = DQNHistory.load(checkpoint_dir / "history.json")
        step = history.steps[-1] if history.steps else 0
        print(f"Resuming from step {step} ({checkpoint_dir})")
    target_net.load_state_dict(q_net.state_dict())

    def save() -> None:
        torch.save(q_net.state_dict(), checkpoint_dir / "q_net.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        history.steps.append(step)
        history.save(checkpoint_dir / "history.json")

    started = time.time() - history.wall_time
    observation, _ = env.reset(seed=config.seed + step)
    episode_return, episode_length = 0.0, 0
    try:
        while step < config.total_steps:
            epsilon = config.epsilon(step)
            if step < config.learning_starts and len(buffer) < config.learning_starts:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(select_action(q_net, observation, epsilon))
            next_observation, reward, terminated, truncated, info = env.step(action)
            buffer.add(observation, action, float(reward), next_observation, terminated)
            observation = next_observation
            episode_return += float(reward)
            episode_length += 1
            step += 1

            if len(buffer) >= config.learning_starts and step % config.train_every == 0:
                batch = buffer.sample(config.batch_size)
                loss = compute_loss(q_net, target_net, batch, config.gamma)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                history.losses.append(float(loss.item()))
                history.loss_steps.append(step)
            if step % config.target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                history.episode_returns.append(episode_return)
                history.episode_lengths.append(episode_length)
                history.episode_end_steps.append(step)
                history.epsilons.append(epsilon)
                history.wall_time = time.time() - started
                if on_episode_end is not None:
                    on_episode_end(history, step)
                observation, _ = env.reset()
                episode_return, episode_length = 0.0, 0
            if step % config.checkpoint_every == 0:
                history.wall_time = time.time() - started
                save()
    except KeyboardInterrupt:
        print("Interrupted: saving a checkpoint, run the cell again to resume.")
    history.wall_time = time.time() - started
    save()
    return history



# ------------------------------------------------------------------------------ deep RL (SAC)


@dataclass
class SACConfig:
    """Hyper-parameters of the SAC training loop (continuous steering)."""

    total_steps: int = 15_000
    learning_starts: int = 1_000
    batch_size: int = 128
    gamma: float = 0.95
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    initial_alpha: float = 0.1
    target_entropy: float = -1.0
    tau: float = 0.01
    train_every: int = 2
    buffer_size: int = 30_000
    augment_pad: int = 2
    checkpoint_every: int = 5_000
    seed: int = 0


@dataclass
class SACHistory(DQNHistory):
    """DQN's record plus the entropy weight after each episode."""

    alphas: list[float] = field(default_factory=list)


def train_sac(  # noqa: PLR0913, PLR0915
    env,
    critic,
    critic_target,
    actor,
    buffer,
    *,
    sac,
    config: SACConfig,
    checkpoint_dir: str | Path,
    on_episode_end: Callable[[SACHistory, int], None] | None = None,
    resume: bool = True,
) -> SACHistory:
    """The SAC training loop.

    Args:
        env: environment from :func:`rl_duckiematrix.env.make_env` with ``actions="steering"``.
        critic, critic_target, actor: the learner's ``Critic`` (twice) and ``Actor``.
        buffer: the learner's continuous-action ``ReplayBuffer``.
        sac: the learner's module (``solution.sac``) providing ``select_action``, ``critic_loss``,
            ``actor_loss``, ``alpha_loss`` and ``random_shift``.
        config: :class:`SACConfig`.
        checkpoint_dir: where ``critic.pt``, ``actor.pt``, ``alpha.pt`` and ``history.json`` go.
        on_episode_end: callback ``(history, step)`` for live plots.
        resume: continue from the checkpoint in ``checkpoint_dir`` when there is one.

    """
    import torch  # noqa: PLC0415

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    log_alpha = torch.tensor(float(np.log(config.initial_alpha)), requires_grad=True)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_learning_rate)
    history = SACHistory()
    step = 0
    if resume and (checkpoint_dir / "actor.pt").exists():
        critic.load_state_dict(torch.load(checkpoint_dir / "critic.pt", map_location="cpu"))
        critic_target.load_state_dict(torch.load(checkpoint_dir / "critic_target.pt", map_location="cpu"))
        actor.load_state_dict(torch.load(checkpoint_dir / "actor.pt", map_location="cpu"))
        with torch.no_grad():
            log_alpha.copy_(torch.load(checkpoint_dir / "alpha.pt", map_location="cpu"))
        history = SACHistory(**json.loads((checkpoint_dir / "history.json").read_text()))
        step = history.steps[-1] if history.steps else 0
        print(f"Resuming from step {step} ({checkpoint_dir})")
    else:
        critic_target.load_state_dict(critic.state_dict())

    def save() -> None:
        torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
        torch.save(critic_target.state_dict(), checkpoint_dir / "critic_target.pt")
        torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
        torch.save(log_alpha.detach(), checkpoint_dir / "alpha.pt")
        history.steps.append(step)
        history.save(checkpoint_dir / "history.json")

    started = time.time() - history.wall_time
    observation, _ = env.reset(seed=config.seed + step)
    episode_return, episode_length = 0.0, 0
    try:
        while step < config.total_steps:
            if len(buffer) < config.learning_starts and step < config.learning_starts:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            else:
                action = sac.select_action(critic, actor, observation, explore=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            buffer.add(observation, action, float(reward), next_observation, terminated)
            observation = next_observation
            episode_return += float(reward)
            episode_length += 1
            step += 1

            if len(buffer) >= config.learning_starts and step % config.train_every == 0:
                batch = buffer.sample(config.batch_size)
                if config.augment_pad:
                    batch = batch._replace(
                        observations=sac.random_shift(batch.observations, config.augment_pad),
                        next_observations=sac.random_shift(batch.next_observations, config.augment_pad),
                    )
                alpha = log_alpha.exp().detach()
                loss = sac.critic_loss(critic, critic_target, actor, batch, config.gamma, alpha)
                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()
                a_loss, log_prob = sac.actor_loss(critic, actor, batch, alpha)
                actor_optimizer.zero_grad()
                a_loss.backward()
                actor_optimizer.step()
                al_loss = sac.alpha_loss(log_alpha, log_prob, config.target_entropy)
                alpha_optimizer.zero_grad()
                al_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    for target, source in zip(critic_target.parameters(), critic.parameters(), strict=True):
                        target.mul_(1.0 - config.tau).add_(config.tau * source)
                history.losses.append(float(loss.item()))
                history.loss_steps.append(step)

            if terminated or truncated:
                history.episode_returns.append(episode_return)
                history.episode_lengths.append(episode_length)
                history.episode_end_steps.append(step)
                history.alphas.append(float(log_alpha.exp().item()))
                history.wall_time = time.time() - started
                if on_episode_end is not None:
                    on_episode_end(history, step)
                observation, _ = env.reset()
                episode_return, episode_length = 0.0, 0
            if step % config.checkpoint_every == 0:
                history.wall_time = time.time() - started
                save()
    except KeyboardInterrupt:
        print("Interrupted: saving a checkpoint, run the cell again to resume.")
    history.wall_time = time.time() - started
    save()
    return history

def evaluate(env, policy: Callable[[np.ndarray], int], *, episodes: int = 5, seed: int = 1000) -> dict[str, Any]:
    """Run ``episodes`` greedy episodes with ``policy(observation) -> action`` and summarize them."""
    returns, lengths, offroad = [], [], 0
    for i in range(episodes):
        observation, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(policy(observation))
            done = terminated or truncated
        stats = info["episode"]
        returns.append(float(stats["r"]))
        lengths.append(int(stats["l"]))
        offroad += int(terminated)
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "offroad_episodes": offroad,
        "returns": returns,
        "lengths": lengths,
    }


def plot_dqn_history(history: DQNHistory, axes=None, window: int = 10):
    """Plot episode returns, episode lengths and the TD loss."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax in axes:
        ax.clear()
    x = history.episode_end_steps
    axes[0].plot(x, history.episode_returns, ".", alpha=0.3, label="episode")
    if len(history.episode_returns) >= window:
        axes[0].plot(x[window - 1:], moving_average(history.episode_returns, window), label=f"mean of {window}")
        axes[0].legend()
    axes[0].set_title("episode return")
    axes[1].plot(x, history.episode_lengths, ".", alpha=0.3)
    if len(history.episode_lengths) >= window:
        axes[1].plot(x[window - 1:], moving_average(history.episode_lengths, window))
    axes[1].set_title("episode length (agent steps)")
    if history.losses:
        axes[2].plot(history.loss_steps, history.losses, alpha=0.5)
        axes[2].set_yscale("log")
    axes[2].set_title("TD loss")
    for ax in axes:
        ax.set_xlabel("environment steps")
    return axes
