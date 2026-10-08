# Exterminator Asteroids

Ambiente de Aprendizado por Reforço que reproduz o **Asteroids** do arcade da Atari
(1979), feito em Python com Pygame e exposto como um ambiente
[Gymnasium](https://gymnasium.farama.org/). O repositório inclui agentes DQN e PPO
(Stable-Baselines3) para treinar e avaliar políticas no jogo.

O ambiente foi auditado e ajustado para seguir as regras do arcade original.

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
git clone https://github.com/ericshantos/exterminator_asteroids.git
cd exterminator_asteroids

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

O modelo é salvo em `models/exterminator_dqn.zip` e as métricas do TensorBoard ficam em
`logs/tensorboard`:

```bash
tensorboard --logdir logs/tensorboard
```

O `DeviceManager` usa a GPU (CUDA) quando ela está disponível e tem memória livre
suficiente. Caso contrário, usa a CPU.

### Treinar o agente PPO

```bash
python -m training.run_ppo
```

Roda 16 ambientes em paralelo (`SubprocVecEnv`). O modelo é
salvo em `models/exterminator_ppo.zip`, junto com as estatísticas de normalização da
recompensa (`models/exterminator_ppo.vecnormalize.pkl`).

### Avaliar um modelo treinado

```bash
python -m training.evaluate              # DQN
python -m training.evaluate --algo ppo   # PPO
python -m training.evaluate --algo ppo --no-render --episodes 50 --seed 0
```

Por padrão, roda 20 episódios com renderização e mostra pontuação, tempo de
sobrevivência, asteroides destruídos e precisão de tiro.

### Jogar uma partida na arena

```bash
python -m arena --algo ppo                       # com janela, em tempo real
python -m arena --algo ppo --no-render --seed 0  # sem janela, o mais rápido possível
python -m arena --model models/outro.zip --algo dqn --output logs/arena/partida.json
```

O pacote `arena` tira as restrições de treino. Não há limite de
`max_episode_steps` nem recompensa, e a partida só termina no game over. O
frame skip continua, porque o modelo foi treinado decidindo a cada 4 frames.
Ao final aparecem:

- a pontuação, a onda alcançada, o tempo de jogo, os pontos por minuto e os
  pontos por vida perdida;
- os asteroides destruídos e os discos destruídos sobre os que apareceram;
- os tiros e a precisão;
- as vidas perdidas e extras, o tempo médio por vida e os hiperespaços;
- as mortes por causa e a distribuição das ações.

`--output` salva o mesmo resultado em JSON.

Opções: `--algo {dqn,ppo}`, `--model CAMINHO`, `--seed N`, `--no-render`,
`--stochastic` (amostra a política em vez de usar a ação mais provável),
`--max-minutes N`, `--max-score N` e `--output ARQUIVO`. Fechar a janela ou apertar Ctrl+C
encerra a partida e mostra as métricas até aquele ponto. O motivo do fim fica
em `end_reason`.

`--max-minutes N` encerra a partida depois de N minutos de jogo (frames ÷ 60,
não tempo real), caso ela não tenha chegado ao game over. O fim fica como
`time_limit`. Como o limite é em tempo de jogo, o resultado é o mesmo com ou
sem janela e com qualquer número de processos. No lote, as partidas que
chegam ao limite entram no resumo junto com as que terminaram em game over, e
o relatório mostra quantas pararam pelo limite.

`--max-score N` encerra a partida quando a pontuação chega a N, com fim
`score_limit`. A checagem acontece a cada decisão do agente, então a pontuação
final pode passar um pouco de N. Os dois limites podem ser usados juntos, e a
partida termina no que vier primeiro. No lote, valem as mesmas regras do limite
de tempo.

#### Avaliação em lote

```bash
python -m arena --algo ppo --no-render --matches 30 --output logs/arena/v0.json
```

Com `--matches N`, a arena joga N partidas com seeds consecutivas, a partir de
`--seed` ou de 1000 por padrão, e mostra a média, o intervalo de confiança de
95%, o desvio, o mínimo e o máximo de cada métrica. Também mostra os pontos por
vida somando todas as partidas e as mortes por causa no lote inteiro.

Sem janela, as partidas rodam em paralelo, com um processo por núcleo; use
`--workers N` para escolher outro número. Cada partida fixa a própria seed, então
o resultado é o mesmo com ou sem paralelismo. Partidas interrompidas por Ctrl+C
ficam de fora do resumo. `--output` salva o resumo e todas as partidas em JSON.

Para comparar duas versões de um modelo, avalie as duas com as mesmas seeds e o
mesmo número de partidas.

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

Aqui cada `step` é um frame. Para usar o mesmo ambiente do treino, com frame skip,
troque `AsteroidEnv(...)` por `make_env(...)`.

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

### Passo do agente e frame skip

O jogo roda a 60 frames por segundo, e cada ação do agente dura
`rl.frame_skip` frames (4 por padrão, ou 15 decisões por segundo). As
recompensas desses frames são somadas. Tiro e hiperespaço valem só no primeiro
frame do bloco, então o agente pode disparar de novo no passo seguinte. A
observação é montada uma vez por passo, no último frame do bloco. Os dois
agentes usam o mesmo ambiente, criado por `env.make_env()`.

### Observação: `Box(-1, 1, shape=(147,), float32)`

Tudo é expresso no **referencial da nave**: o eixo "proa" aponta para onde a nave
mira e o eixo "direita" para o lado direito dela. Distâncias são toroidais.

| Índices | Conteúdo |
|---|---|
| 0–6 | Nave: velocidade (proa, direita), viva, no hiperespaço, tiros disponíveis (de 4), vidas, número de asteroides |
| 7–19 | Disco voador: 12 features de alvo + 1 se for o disco pequeno (zeros sem disco) |
| 20–31 | Até 2 tiros do disco: presente, posição (proa, direita), velocidade relativa (proa, direita), tempo até a colisão |
| 32–127 | Os 8 asteroides mais próximos (12 features cada), ordenados pela distância entre as bordas |
| 128–146 | Radar de fuga: 16 fatias + 3 de resumo (abaixo) |

Features de cada alvo:

| # | Feature |
|---|---|
| 0 | Presente (1) ou vaga vazia (0) |
| 1–2 | Posição relativa (proa, direita) |
| 3 | Distância entre as bordas (0 = encostando) |
| 4–5 | Velocidade relativa (proa, direita) |
| 6 | Tamanho |
| 7 | Tempo até a colisão em linha reta, normalizado por 2 s (1 = não colide) |
| 8 | Distância de passagem: quão perto o alvo vai passar se nada mudar |
| 9 | Ângulo de interceptação: para onde atirar, já considerando o movimento do alvo e a velocidade do projétil |
| 10 | Interceptável: o projétil alcança o alvo antes de expirar |
| 11 | Acerta se atirar agora |

O radar de fuga responde, para 16 direções a partir do nariz (22,5° cada, em
sentido horário: 0°, 22,5° … 180°, −157,5° … −22,5°), quanto tempo a nave tem
até bater em algo se fugir para lá. A fuga tem dois trechos: a nave gira até a
direção, seguindo o curso atual enquanto gira (4,22° por frame), e depois ganha
3 px/frame naquela direção, sem passar da velocidade máxima. O radar considera
todos os asteroides, o disco e os tiros do disco, não só os 8 mais próximos.

| Índices | Conteúdo |
|---|---|
| 128–143 | Tempo até a colisão fugindo para cada fatia, normalizado por 2 s (1 = livre) |
| 144 | Curso atual: tempo até a colisão sem fugir, entre todos os objetos |
| 145 | Ângulo da melhor fatia ÷ 180° (positivo = direita); no empate, a mais perto do nariz |
| 146 | Valor da melhor fatia |

A geometria (eixos da nave, tempo até a colisão, interceptação, fuga) fica em
`env/kinematics.py`. A convenção de ângulo é a mesma em todo o projeto: 0 aponta
para cima e cresce no sentido horário, com y crescendo para baixo.

### Recompensa

A recompensa (`env/reward.py`) é calculada por frame, e os pesos ficam na seção
`reward` do `environment.yaml`:

| Termo | Chave | Padrão |
|---|---|---|
| Pontos do jogo × escala (paga os acertos) | `score_scale` | 0,01 |
| Por frame com a nave ativa (viva e fora do hiperespaço) | `survive_step` | 0,002 |
| Vida perdida | `life_lost` | −3,0 |
| Fim do jogo | `game_over` | −10,0 |
| Entrada no hiperespaço | `hyperspace_cost` | −0,05 |
| Projétil que expira sem acertar nada | `missed_shot` | −0,02 |
| Shaping de ameaça: diferença de potencial com Φ = −ameaça | `danger_shaping` | 0,5 |
| Shaping de mira: variação do alinhamento com o ângulo de interceptação | `aim_shaping` | 0,5 |

A ameaça vai de 0 a 1 e mede quão iminente é a colisão mais próxima (asteroides,
disco e tiros do disco) num horizonte de 2 s. Como o shaping é uma diferença de
potenciais (Ng et al., 1999), ele adianta o sinal da morte sem mudar a política
ótima.

O shaping de ameaça só é pago entre dois frames seguidos com a nave ativa (viva e
fora do hiperespaço). Entrar no hiperespaço ou morrer não rende o crédito de
"perigo resolvido", e reaparecer num lugar perigoso, depois do hiperespaço ou do
renascimento, não é cobrado, porque o lugar é sorteado. Assim, só um desvio de
verdade recupera o que a ameaça custou, e o valor do hiperespaço vem apenas das
consequências reais: o custo de entrada, o tempo sem bônus de sobrevivência e o
risco de explodir na volta.

O shaping de mira também é uma diferença: paga quando a nave gira em direção ao
ponto de interceptação do alvo que o projétil alcança mais rápido e cobra quando
ela se afasta. Trocar de alvo (inclusive ao destruí-lo) não gera recompensa. Por
isso, girar sem parar não acumula nada, e o acerto em si é pago pela pontuação.

### Término do episódio

- `terminated`: as vidas acabaram.
- `truncated`: o episódio chegou a `rl.max_episode_steps` frames.

O `info` de cada passo traz `score`, `lives`, `wave`, `frame_count`, `asteroid_destroyed`,
`shots_fired`, `accuracy_hits` e `accuracy`, além das métricas de diagnóstico:

| Chave | Conteúdo |
|---|---|
| `lives_lost` | Vidas perdidas no episódio |
| `points_per_life` | Pontuação dividida pelas vidas perdidas (a pontuação, se nenhuma foi perdida) |
| `saucers_spawned`, `saucers_destroyed` | Discos que apareceram e discos destruídos pela nave |
| `hyperspace_jumps` | Entradas no hiperespaço |
| `deaths_<causa>` | Mortes por causa: `asteroid`, `hyperspace_landing` (asteroide até 1 s depois de sair do hiperespaço), `hyperspace_failure`, `saucer_bullet` e `saucer_collision` |

O `ExterminatorMetricsCallback` envia essas métricas ao TensorBoard no fim de cada
episódio, com o prefixo `exterminator/`.

## Configuração

Os arquivos YAML em `configs/yaml/` são carregados e validados pelos dataclasses
de `configs/schema.py`.

| Arquivo | Seções | Principais chaves |
|---|---|---|
| `environment.yaml` | `screen` | `width`, `height`, `fps`, `window_title` |
| | `game` | `starting_lives`, `extra_life_score`, `initial_asteroids`, `max_asteroids` |
| | `reward` | `survive_step`, `score_scale`, `life_lost`, `game_over`, `hyperspace_cost`, `missed_shot`, `danger_shaping`, `aim_shaping` |
| | `rl` | `max_episode_steps` (em frames), `frame_skip` |
| `dqn.yaml` | `dqn` | `learning_rate`, `gamma`, `buffer_size`, `batch_size`, `learning_starts`, `train_freq`, `gradient_steps`, `target_update_interval`, `exploration`, `total_timesteps` |
| `training.yaml` | `training` | `total_timesteps`, `seed`, `device` |
| `ppo.yaml` | `ppo` | `n_envs`, `learning_rate` (com decaimento linear), `gamma`, `gae_lambda`, `clip_range`, `n_steps`, `batch_size`, `epochs`, `ent_coef`, `total_timesteps` |

A configuração é carregada uma vez, na importação, e fica disponível em
`configs.cfg`.

## Estrutura do projeto

```
exterminator_asteroids/
├── agent/
│   ├── agents/dqn_agent.py        # DQN (Stable-Baselines3)
│   ├── agents/ppo_agent.py        # PPO com ambientes paralelos e frame skip
│   ├── callbacks/                 # métricas do jogo no TensorBoard
│   └── protocols/                 # contrato comum dos agentes
├── arena/                        # partida completa até o game over, com métricas
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
│   ├── kinematics.py              # geometria no referencial da nave
│   ├── reward.py                  # função de recompensa
│   ├── frame_skip.py              # wrapper de frame skip
│   └── factory.py                 # make_env(): ambiente de treino e avaliação
├── rendering/                     # renderer, HUD e fonte
├── training/                      # treino, avaliação, DeviceManager
├── tests/                         # testes de cinemática, ambiente e arena
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
