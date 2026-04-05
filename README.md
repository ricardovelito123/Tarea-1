# Tensor++

Librería de tensores implementada en C++ como parte de la Tarea #1 del curso **Programación III** (2026-1), inspirada en bibliotecas científicas como NumPy y PyTorch.

---

## Descripción

La librería implementa una clase `Tensor` capaz de manejar tensores de hasta 3 dimensiones, con soporte para operaciones matemáticas, transformaciones no lineales y gestión manual de memoria dinámica. El programa principal demuestra su uso construyendo el forward pass de una red neuronal de dos capas densas.

---

## Estructura del proyecto

```
.
└── main.cpp    # Código fuente completo: clase Tensor, transformaciones y red neuronal
```

---

## Compilación

### Requisitos

- Compilador C++ con soporte para **C++11 o superior** (g++, clang++, MSVC)

### Con g++ (Linux / macOS / MinGW en Windows)

```bash
g++ -std=c++11 -o tensor main.cpp
```

### Con CMake (como en CLion)

Si usas CLion u otro IDE con CMake, el proyecto se compila automáticamente al ejecutar desde el IDE. El ejecutable queda en `cmake-build-debug/`.

---

## Ejecución

```bash
./tensor
```

En Windows:

```bash
tensor.exe
```

### Salida esperada

```
Paso 1 - Entrada:        {1000, 20, 20} -> total: 400000
Paso 2 - view:           {1000, 400}    -> total: 400000
Paso 3 - matmul W1:      {1000, 100}    -> total: 100000
Paso 4 - suma bias b1:   {1000, 100}    -> total: 100000
Paso 5 - ReLU:           {1000, 100}    -> total: 100000
Paso 6 - matmul W2:      {1000, 10}     -> total: 10000
Paso 7 - suma bias b2:   {1000, 10}     -> total: 10000
Paso 8 - Sigmoid:        {1000, 10}     -> total: 10000

Primeras 10 salidas de la muestra 0:
  clase[0] = 0.552481
  ...
```

Los valores de salida varían en cada ejecución ya que los pesos y la entrada se generan con `Tensor::random`.

---

## Funcionalidades implementadas

### Clase `Tensor`

| Sección | Funcionalidad |
|--------|--------------|
| 3.1 | Constructor principal con `shape` y `values` |
| 3.2 | Métodos estáticos: `zeros`, `ones`, `random`, `arange` |
| 4 | Constructor de copia, constructor de movimiento, `operator=` copia y movimiento, destructor |
| 5 | Interfaz `TensorTransform` y método `apply` con polimorfismo en tiempo de ejecución |
| 6 | Operadores `+`, `-`, `*` (elemento a elemento) y `*` por escalar |
| 7 | `view` para reinterpretar forma y `unsqueeze` para insertar dimensiones |
| 8 | `concat` estático para unir tensores por una dimensión |
| 9 | Funciones amigas `dot` y `matmul` |
| 10 | Red neuronal de dos capas densas con ReLU y Sigmoid |

### Transformaciones (`TensorTransform`)

- **ReLU**: `y = max(0, x)` — elimina valores negativos.
- **Sigmoid**: `y = 1 / (1 + e^-x)` — comprime valores al rango (0, 1).

### Gestión de memoria

Los datos se almacenan en un **array dinámico contiguo** (`double*`). El ciclo de vida del tensor sigue la regla de los cinco (Rule of Five): constructor de copia, constructor de movimiento, asignación de copia, asignación de movimiento y destructor.

---

## Red neuronal (Sección 10)

El `main` implementa el forward pass de una red neuronal con la siguiente arquitectura:

| Paso | Operación | Dimensión resultante |
|------|-----------|----------------------|
| 1 | Tensor de entrada aleatorio | 1000 × 20 × 20 |
| 2 | `view` | 1000 × 400 |
| 3 | `matmul` con W1 (400 × 100) | 1000 × 100 |
| 4 | Suma con bias b1 | 1000 × 100 |
| 5 | Activación ReLU | 1000 × 100 |
| 6 | `matmul` con W2 (100 × 10) | 1000 × 10 |
| 7 | Suma con bias b2 | 1000 × 10 |
| 8 | Activación Sigmoid | 1000 × 10 |

---

## Autores

- Humberto Ricardo Velito Neira
- 

Curso: Programación III — UTEC, 2026-1
