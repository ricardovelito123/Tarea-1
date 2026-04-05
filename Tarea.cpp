#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
using namespace std;

class TensorTransform;

class Tensor {
    size_t total;      // cantidad total de elementos del tensor
    double *values;    // puntero al bloque de memoria donde se guardan los datos
    vector<size_t> shape; // dimensiones del tensor, ej: {2,3} para una matriz 2x3

    // Damos acceso a los atributos privados a estas clases y funciones
    friend class ReLU;
    friend class Sigmoid;

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

public:
    // sección 3.1: Constructor principal
    // recibe las dimensiones y los valores, reserva memoria dinámica y copia los datos
    Tensor(const vector<size_t> &shape, const vector<double> &vals) {
        this->shape = shape;
        total = 1;
        for (size_t i = 0; i < shape.size(); i++) total *= shape[i]; // calcula total multiplicando dimensiones

        if (vals.size() != total)
            throw invalid_argument("Valores no coinciden con shape");

        values = new double[total]; // reserva memoria dinámica para todos los elementos
        for (size_t i = 0; i < total; i++) values[i] = vals[i]; // copia los valores al bloque de memoria
    }

    // Getters para acceder a los atributos privados desde el main
    size_t getTotal() const { return total; }
    double* getValues() const { return values; }

    // Sección 3.2: Métodos estáticos para crear tensores predefinidos
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    // Sección 5.2: Método que delega la transformación a la clase recibida (polimorfismo)
    Tensor apply(const TensorTransform& transform) const;

    // Sección 4: Constructor de copia (deep copy)
    // Crea un tensor nuevo con su propia memoria, copiando todos los datos del original
    Tensor(const Tensor& other) {
        total = other.total;
        shape = other.shape;
        values = new double[total]; // nueva memoria independiente
        for (size_t i = 0; i < total; i++) values[i] = other.values[i]; // copia elemento a elemento
    }

    // Sección 4: Destructor
    // Libera la memoria dinámica reservada al crearse el tensor
    ~Tensor() { delete[] values; }

    // sección 4: Asignación de copia (operator=)
    // libera la memoria actual, luego copia los datos del otro tensor
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this; // evita auto-asignación
        delete[] values; // libera memoria actual antes de copiar
        total = other.total;
        shape = other.shape;
        values = new double[total];
        for (size_t i = 0; i < total; i++) values[i] = other.values[i];
        return *this;
    }

    // Sección 4: Constructor de movimiento
    // Transfiere la propiedad del puntero al nuevo objeto, sin copiar datos
    Tensor(Tensor&& other) noexcept {
        total = other.total;
        shape = move(other.shape); // transfiere el vector sin copiar
        values = other.values;     // roba el puntero
        other.values = nullptr;    // deja al origen sin puntero (estado valido pero vacío)
        other.total = 0;
    }

    // sección 4: Asignación de movimiento (operator=)
    // Libera recursos actuales y toma posesión de los del objeto temporal
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        delete[] values; // libera lo que tenía antes
        total = other.total;
        shape = move(other.shape);
        values = other.values;     // roba el puntero sin copiar datos
        other.values = nullptr;    // deja al origen en estado válido pero nulo
        other.total = 0;
        return *this;
    }

    // sección 6: Sobrecarga de operadores aritmeticos

    // Suma elemento a elemento, requiere que ambos tensores tengan la misma forma
    Tensor operator+(const Tensor& other) const {
        if (shape != other.shape)
            throw invalid_argument("Suma incompatible");

        vector<double> result;
        for (size_t i = 0; i < total; i++)
            result.push_back(values[i] + other.values[i]); // suma posición a posición

        return Tensor(shape, result);
    }

    // resta elemento a elemento, requiere que ambos tensores tengan la misma forma
    Tensor operator-(const Tensor& other) const {
        if (shape != other.shape)
            throw invalid_argument("Resta incompatible");

        vector<double> result;
        for (size_t i = 0; i < total; i++)
            result.push_back(values[i] - other.values[i]); // resta posición a posición

        return Tensor(shape, result);
    }

    // Multiplicación elemento a elemento (no es matmul), requiere misma forma
    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape)
            throw invalid_argument("Multiplicacion incompatible");

        vector<double> result;
        for (size_t i = 0; i < total; i++)
            result.push_back(values[i] * other.values[i]); // multiplica posición a posición

        return Tensor(shape, result);
    }

    // multiplicación por escalar: multiplica cada elemento por un número
    Tensor operator*(double scalar) const {
        vector<double> result;
        for (size_t i = 0; i < total; i++)
            result.push_back(values[i] * scalar); // escala cada elemento

        return Tensor(shape, result);
    }

    // sección 7.1: view
    // cambia la forma lógica del tensor manteniendo la misma cantidad total de elementos
    Tensor view(const vector<size_t>& new_shape) const {
        size_t new_total = 1;
        for (size_t i = 0; i < new_shape.size(); i++) new_total *= new_shape[i];

        if (new_total != total)
            throw invalid_argument("View incompatible");

        if (new_shape.size() > 3)
            throw invalid_argument("Maximo 3 dimensiones");

        vector<double> vals;
        for (size_t i = 0; i < total; i++) vals.push_back(values[i]);

        return Tensor(new_shape, vals);
    }

    // sección 7.2: unsqueeze
    // inserta una dimensión de tamaño 1 en la posición indicada
    Tensor unsqueeze(size_t dim) const {
        if (shape.size() >= 3)
            throw invalid_argument("Maximo 3 dimensiones");

        if (dim > shape.size())
            throw invalid_argument("Dimension invalida");

        vector<size_t> new_shape = shape;
        new_shape.insert(new_shape.begin() + dim, 1);

        vector<double> vals;
        for (size_t i = 0; i < total; i++) vals.push_back(values[i]);

        return Tensor(new_shape, vals);
    }

    // sección 8: concat
    // concatena varios tensores a lo largo de una dimensión específica
    static Tensor concat(const vector<Tensor>& tensors, size_t dim);
};

// FUNCIONES AMIGAS
//  dot: producto punto entre dos tensores de igual número de elementos
// Multiplica elemento a elemento y suma todo → resultado es un escalar en tensor {1}
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.total != b.total)
        throw invalid_argument("dot incompatible");

    double sum = 0;
    for (size_t i = 0; i < a.total; i++)
        sum += a.values[i] * b.values[i]; // acumula productos

    return move(Tensor({1}, {sum})); // retorna tensor escalar
}

// matmul: multiplicación matricial entre tensores 2D
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw invalid_argument("matmul solo 2D");

    size_t m = a.shape[0]; // filas de A
    size_t n = a.shape[1]; // columnas de A = filas de B
    size_t p = b.shape[1]; // columnas de B

    if (n != b.shape[0])
        throw invalid_argument("matmul incompatible"); // columnas de A deben ser iguales a filas de B

    vector<double> result(m * p, 0.0); // inicializa resultado en cero

    // triple loop multiplicacion matricial
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < p; j++)
            for (size_t k = 0; k < n; k++)
                result[i * p + j] += a.values[i * n + k] * b.values[k * p + j];
    return move(Tensor({m, p}, result));
}

// (3.2) MÉTODOS ESTÁTICOS
// Crea tensor lleno de ceros
Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    return Tensor(shape, vector<double>(total, 0));
}

// Crea tensor lleno de unos
Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    return Tensor(shape, vector<double>(total, 1));
}

// Crea tensor con valores aleatorios en el rango [min, max)
Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    size_t total = 1;
    for (auto s : shape) total *= s;

    vector<double> vals;
    for (size_t i = 0; i < total; i++)
        vals.push_back(min + (double)rand() / RAND_MAX * (max - min));

    return Tensor(shape, vals);
}

// Crea tensor 1D con valores secuenciales desde start hasta end (no inclusivo)
Tensor Tensor::arange(int start, int end) {
    vector<double> vals;
    for (int i = start; i < end; i++) vals.push_back(i);
    return Tensor({(size_t)(end - start)}, vals);
}

// Seccion(5) TRANSFORMACIONES

// interfaz abstracta = cualquier clase que herede debe implementar apply
class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

// ReLU: reemplaza valores negativos por 0, deja los positivos igual
// Formula: y = max(0, x)
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        vector<double> v;
        for (size_t i = 0; i < t.total; i++)
            v.push_back(max(0.0, t.values[i]));
        return Tensor(t.shape, v);
    }
};

// Sigmoid: comprime los valores al rango (0, 1)
// Fórmula: y = 1 / (1 + e^-x)
class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        vector<double> v;
        for (size_t i = 0; i < t.total; i++)
            v.push_back(1.0 / (1.0 + exp(-t.values[i])));
        return Tensor(t.shape, v);
    }
};

// Delegación del apply: el tensor le pasa su propio contenido a la transformación recibida
Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

// sección 8: concat
// concatena tensores compatibles a lo largo de una dimensión
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty())
        throw invalid_argument("No hay tensores para concatenar");

    vector<size_t> base_shape = tensors[0].shape;

    if (dim >= base_shape.size())
        throw invalid_argument("Dimension invalida");

    if (base_shape.size() > 3)
        throw invalid_argument("Maximo 3 dimensiones");

    // validar misma cantidad de dimensiones y compatibilidad
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i].shape.size() != base_shape.size())
            throw invalid_argument("Concat incompatible");

        for (size_t j = 0; j < base_shape.size(); j++) {
            if (j != dim && tensors[i].shape[j] != base_shape[j])
                throw invalid_argument("Concat incompatible");
        }
    }

    vector<size_t> new_shape = base_shape;
    new_shape[dim] = 0;
    for (size_t i = 0; i < tensors.size(); i++)
        new_shape[dim] += tensors[i].shape[dim];

    vector<double> result;

    // caso 1D
    if (base_shape.size() == 1) {
        for (size_t i = 0; i < tensors.size(); i++) {
            for (size_t j = 0; j < tensors[i].total; j++)
                result.push_back(tensors[i].values[j]);
        }
    }

    // caso 2D
    else if (base_shape.size() == 2) {
        if (dim == 0) {
            for (size_t t = 0; t < tensors.size(); t++) {
                for (size_t i = 0; i < tensors[t].total; i++)
                    result.push_back(tensors[t].values[i]);
            }
        } else {
            for (size_t i = 0; i < base_shape[0]; i++) {
                for (size_t t = 0; t < tensors.size(); t++) {
                    size_t current_cols = tensors[t].shape[1];
                    for (size_t j = 0; j < current_cols; j++) {
                        result.push_back(tensors[t].values[i * current_cols + j]);
                    }
                }
            }
        }
    }

    // caso 3D
    else if (base_shape.size() == 3) {
        size_t a = base_shape[0];
        size_t b = base_shape[1];

        if (dim == 0) {
            for (size_t t = 0; t < tensors.size(); t++) {
                for (size_t i = 0; i < tensors[t].total; i++)
                    result.push_back(tensors[t].values[i]);
            }
        } else if (dim == 1) {
            for (size_t i = 0; i < a; i++) {
                for (size_t t = 0; t < tensors.size(); t++) {
                    size_t cur_b = tensors[t].shape[1];
                    size_t cur_c = tensors[t].shape[2];
                    for (size_t j = 0; j < cur_b; j++) {
                        for (size_t k = 0; k < cur_c; k++) {
                            size_t pos = i * (cur_b * cur_c) + j * cur_c + k;
                            result.push_back(tensors[t].values[pos]);
                        }
                    }
                }
            }
        } else if (dim == 2) {
            for (size_t i = 0; i < a; i++) {
                for (size_t j = 0; j < b; j++) {
                    for (size_t t = 0; t < tensors.size(); t++) {
                        size_t cur_c = tensors[t].shape[2];
                        for (size_t k = 0; k < cur_c; k++) {
                            size_t pos = i * (tensors[t].shape[1] * cur_c) + j * cur_c + k;
                            result.push_back(tensors[t].values[pos]);
                        }
                    }
                }
            }
        }
    }

    return Tensor(new_shape, result);
}
