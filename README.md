# Projeto Apollo

Ambiente de Aprendizado por Reforço que reproduz o **Asteroids** do arcade da Atari
(1979), feito em Python com Pygame e exposto como um ambiente
[Gymnasium](https://gymnasium.farama.org/). O repositório inclui um agente DQN
(Stable-Baselines3) para treinar e avaliar políticas no jogo.

O ambiente foi auditado e ajustado para seguir as regras do arcade original. Os
detalhes estão em [`RELATORIO_PROJETO_APOLLO_ASTEROIDS_1979.pdf`](RELATORIO_PROJETO_APOLLO_ASTEROIDS_1979.pdf)
e em [`RELATORIO_FIDELIDADE_ASTEROIDS_1979.txt`](RELATORIO_FIDELIDADE_ASTEROIDS_1979.txt).

## Sumário

- [Requisitos e instalação](#requisitos-e-instalação)
- [Uso rápido](#uso-rápido)
- [Regras do jogo](#regras-do-jogo)
- [Interface de RL](#interface-de-rl)
- [Configuração](#configuração)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Desenvolvimento](#desenvolvimento)
- [Licença](#licença)

## Requisitos e instalação

- Python 3.10 ou superior
- Dependências: `numpy`, `pygame`, `gymnasium`, `stable-baselines3`, `dacite` e
  `pyyaml` (o PyTorch é instalado junto com o Stable-Baselines3)

```bash
git clone <url-do-repositorio> project-apollo
cd project-apollo

python -m venv .venv
source .venv/bin/activate

pip install numpy pygame gymnasium stable-baselines3 dacite pyyaml
```

Para as ferramentas de desenvolvimento:

```bash
pip install black ruff mypy pytest pre-commit types-PyYAML
```

> O `pyproject.toml` ainda não declara quais pacotes distribuir. Por isso
> `pip install -e .` falha com *"Multiple top-level packages discovered in a
> flat-layout"*. Instale as dependências diretamente, como acima.

> Rode os comandos a partir da raiz do repositório: a fonte do HUD
> (`assets/fonts/Hyperspace.otf`) e as pastas `models/` e `logs/` usam caminhos
> relativos.

## Uso rápido

### Treinar o agente DQN

```bash
python -m training.run_dqn
```

O modelo é salvo em `models/apollo_dqn.zip` e as métricas do TensorBoard ficam em
`logs/tensorboard`:

```bash
tensorboard --logdir logs/tensorboard
```

O `DeviceManager` usa a GPU (CUDA) quando ela está disponível e tem memória livre
suficiente. Caso contrário, usa a CPU.

### Avaliar um modelo treinado

```bash
python -m training.evaluate
```

Roda 20 episódios com renderização e mostra pontuação, asteroides destruídos e
precisão de tiro.

### Usar o ambiente diretamente

```python
from env import AsteroidEnv

env = AsteroidEnv(render_mode="human")  # ou None para treino sem janela
obs, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

Sem janela (servidores, CI), defina `SDL_VIDEODRIVER=dummy`.

## Regras do jogo

A referência é o **arcade vetorial de 1979**, e não a versão para o Atari 2600
(1981), que tem regras diferentes. O jogo roda a 60 passos por segundo, e cada
`step` do ambiente equivale a um frame.

| Elemento | Comportamento |
|---|---|
| Campo de jogo | 1024×768 (4:3), toroidal: tudo que sai por uma borda volta pela oposta |
| Nave | Gira 3/256 de volta por frame. Tem inércia, e o arrasto só age sem propulsão. Pode girar e acelerar ao mesmo tempo |
| Tiro | Até 4 tiros na tela. Um tiro por aperto do botão (sem tiro automático). Sai do nariz e soma a velocidade da nave |
| Hiperespaço | Um uso por aperto. A nave some por 48 frames e reaparece num ponto aleatório. Na reentrada, a chance de explodir cresce com o número de asteroides |
| Asteroides | 3 tamanhos na razão 4:2:1, valendo 20, 50 e 100 pontos. Cada um se divide em 2 do tamanho menor, e os fragmentos herdam o movimento do pai. Limite de 26 na tela |
| Ondas | 4, 6, 8, 10, 11, 11… asteroides, surgindo nas bordas, com uma pausa entre ondas |
| Discos voadores | Grande (200 pts) atira em direções aleatórias. Pequeno (1000 pts) mira no jogador, com precisão que cresce com a pontuação. Até 2 tiros. A partir de 40.000 pts só aparecem discos pequenos |
| Colisões | Nave × asteroide e nave × disco destroem os dois e dão os pontos. Disco × asteroide destrói os dois. Tiros do disco destroem asteroides, sem pontos |
| Vidas | 3 iniciais e uma extra a cada 10.000 pts, sem limite. A nave só reaparece com o centro livre |
| Placar | O valor exibido volta a zero ao chegar em 100.000. O valor interno é preservado |

Algumas constantes não têm valor de consenso documentado e foram aproximadas: a
duração e a chance de falha do hiperespaço, o intervalo de tiro do disco e as
velocidades dos asteroides. Elas estão indicadas no relatório.

## Interface de RL

### Ações: `Discrete(13)`

| ID | Ação | ID | Ação |
|---|---|---|---|
| 0 | Nada | 7 | Esquerda + propulsão |
| 1 | Girar à esquerda | 8 | Direita + propulsão |
| 2 | Girar à direita | 9 | Esquerda + tiro |
| 3 | Propulsão | 10 | Direita + tiro |
| 4 | Tiro | 11 | Esquerda + propulsão + tiro |
| 5 | Hiperespaço | 12 | Direita + propulsão + tiro |
| 6 | Propulsão + tiro | | |

Tiro e hiperespaço só são ativados na borda de subida. Para disparar de novo, o
agente precisa escolher uma ação sem tiro em pelo menos um frame.

### Observação: `Box(-1, 1, shape=(74,), float32)`

| Índices | Conteúdo |
|---|---|
| 0–3 | Nave: `vx`, `vy` normalizados, `cos` e `sin` do ângulo |
| 4–14 | Disco voador (11 features de alvo; zeros se não houver disco) |
| 15 | 1 se há disco voador, senão 0 |
| 16–70 | Os 5 asteroides mais perigosos (11 features cada; zeros nas vagas vazias) |
| 71–73 | Risco total e vetor de risco (x, y) |

Features de cada alvo, todas relativas à nave e com distância toroidal:
distância normalizada, `sin`/`cos` do ângulo em relação à proa, desvio lateral da
linha de tiro, indicador "na linha de tiro", velocidade relativa (x, y), tamanho,
velocidade de aproximação, tempo até a colisão e índice de perigo.

Todas as entidades usam a mesma convenção: ângulo 0 aponta para cima e cresce no
sentido horário, e as coordenadas são as da tela (y cresce para baixo).

### Recompensa

A recompensa (`env/reward.py`) combina:

- bônus por passo sobrevivido (`reward.survive_step`);
- +0,01 por ponto marcado no jogo;
- −3,0 por vida perdida e −10,0 no fim do jogo;
- pequenos custos por ação (rotação, propulsão, tiro, hiperespaço);
- penalidade proporcional ao risco atual e bônus quando o risco diminui;
- bônus de mira (alinhamento da proa com o disco ou com o asteroide mais
  próximo) e bônus por tiro bem alinhado, dado só quando um projétil sai de fato.

### Término do episódio

- `terminated`: as vidas acabaram.
- `truncated`: o episódio chegou a `rl.max_episode_steps` passos.

O `info` de cada passo traz `score`, `wave`, `frame_count`, `asteroid_destroyed`,
`shots_fired`, `accuracy_hits` e `accuracy`.

## Configuração

Os arquivos YAML em `configs/yaml/` são carregados e validados pelos dataclasses
de `configs/schema.py`.

| Arquivo | Seções | Principais chaves |
|---|---|---|
| `environment.yaml` | `screen` | `width`, `height`, `fps`, `window_title` |
| | `game` | `starting_lives`, `extra_life_score`, `initial_asteroids`, `max_asteroids` |
| | `reward` | `survive_step` |
| | `rl` | `max_episode_steps` |
| `dqn.yaml` | `dqn` | `learning_rate`, `gamma`, `buffer_size`, `batch_size`, `learning_starts`, `target_update_interval`, `exploration`, `total_timesteps` |
| `training.yaml` | `training` | `total_timesteps`, `seed`, `device` |
| `ppo.yaml` | `ppo` | Hiperparâmetros reservados para um agente PPO (ainda não implementado) |

A configuração é carregada uma vez, na importação, e fica disponível em
`configs.cfg`.

## Estrutura do projeto

```
project-apollo/
├── agent/
│   ├── agents/dqn_agent.py        # DQN (Stable-Baselines3)
│   ├── callbacks/                 # métricas do jogo no TensorBoard
│   └── protocols/                 # contrato comum dos agentes
├── configs/
│   ├── yaml/                      # environment, dqn, ppo, training
│   ├── schema.py                  # dataclasses da configuração
│   └── config_loader.py
├── entities/
│   ├── asteroid/                  # Asteroid e AsteroidManager
│   ├── bullet/                    # Bullet e BulletManager
│   └── ship/
│       ├── player/                # nave, hiperespaço, explosão
│       └── saucer/                # disco voador e seu gerenciador
├── env/
│   ├── asteroid_env.py            # ambiente Gymnasium
│   ├── game_world.py              # laço do jogo, ondas, respawn
│   ├── collision_manager.py       # regras de colisão
│   ├── score_manager.py           # pontuação e vidas extras
│   ├── toroidal_space.py          # geometria toroidal
│   ├── action_space.py            # mapeamento das 13 ações
│   ├── observation.py             # vetor de observação
│   ├── danger_model.py            # ranking de risco dos asteroides
│   └── reward.py                  # função de recompensa
├── rendering/                     # renderer, HUD e fonte
├── training/                      # treino, avaliação, DeviceManager
└── assets/fonts/Hyperspace.otf
```

O `GameWorld` guarda o estado do jogo e chama, a cada frame: nave, asteroides,
disco, colisões, respawn e ondas. O `AsteroidEnv` encapsula o `GameWorld` e
monta observação, recompensa e `info`. A renderização só é criada com
`render_mode="human"`.

## Desenvolvimento

O projeto usa black, ruff, mypy (modo estrito) e hooks de pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```

O pytest está configurado para procurar testes em `tests/`.

## Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE).
