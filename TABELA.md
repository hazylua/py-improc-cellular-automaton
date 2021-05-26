## Resultados

Analisando um número de regras `r = 16`, temos os seguintes resultados, em que `n` indica o número de processos/threads utilizados para alcançar o tempo:

| Paralelização     | Tempo: n = 4 | Tempo: n = 1 | Speedup | Eficiência |
|-------------------|--------------|--------------|---------|------------|
| `multiprocessing` | 23,7167 s    | 83,5647 s    | 3,5234  | 0,8808     |
| `pymp`            | 23,4395 s    | 79,7337 s    | 3,4016  | 0,8504     |
| `threading`*      | -            | 83,4367 s    | -       | -          |

<sup>*`threading` do python não consegue executar mais de uma thread por vez devido ao GIL (Global Interpreter Lock), o qual trava o interpretador do Python em uma única thread.</sup>