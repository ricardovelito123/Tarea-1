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
        for (size_t i = 0; i < new_shape.size(); i++) new_total *= new_shape[i]; // calcula cuántos elementos tendría la nueva forma

        if (new_total != total)
            throw invalid_argument("View incompatible");  // la nueva forma debe tener exactamente el mismo número de elementos

        if (new_shape.size() > 3)
            throw invalid_argument("Maximo 3 dimensiones"); // restricción del enunciado: máximo 3D

        vector<double> vals;
        for (size_t i = 0; i < total; i++) vals.push_back(values[i]);

        return Tensor(new_shape, vals); // crea un nuevo tensor con los mismos datos pero distinta forma
    }

    // sección 7.2: unsqueeze
    // inserta una dimensión de tamaño 1 en la posición indicada
    Tensor unsqueeze(size_t dim) const {
        if (shape.size() >= 3)
            throw invalid_argument("Maximo 3 dimensiones");

        if (dim > shape.size())
            throw invalid_argument("Dimension invalida");

        vector<size_t> new_shape = shape; // copia la forma actual para modificarla
        new_shape.insert(new_shape.begin() + dim, 1);  // inserta un 1 en la posición dim, ej: {3} con dim=0 → {1,3}
        vector<double> vals;
        for (size_t i = 0; i < total; i++) vals.push_back(values[i]); // los datos no cambian, solo la forma

        return Tensor(new_shape, vals); // nuevo tensor con dimensión extra insertada
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
        throw invalid_argument("dot incompatible"); // ambos tensores deben tener el mismo número de elementos

    double sum = 0;
    for (size_t i = 0; i < a.total; i++)
        sum += a.values[i] * b.values[i]; // multiplica cada par y acumula en sum

    return move(Tensor({1}, {sum}));  // empaqueta el escalar resultante en un tensor de una sola posición
}

// matmul: multiplicación matricial entre tensores 2D
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw invalid_argument("matmul solo 2D"); // solo funciona con matrices, no con 1D o 3D

    size_t m = a.shape[0]; // filas de A
    size_t n = a.shape[1]; // columnas de A = filas de B
    size_t p = b.shape[1]; // columnas de B

    if (n != b.shape[0])
        throw invalid_argument("matmul incompatible"); // columnas de A deben ser iguales a filas de B

    vector<double> result(m * p, 0.0); // inicializa resultado en cero

    // triple loop multiplicacion matricial
    for (size_t i = 0; i < m; i++) // recorre filas de A
        for (size_t j = 0; j < p; j++) // recorre columnas de B
            for (size_t k = 0; k < n; k++)  // recorre la dimensión compartida
                 // i*n+k: posición de A[i][k] en memoria lineal
                 // k*p+j: posición de B[k][j] en memoria lineal
                 // i*p+j: posición de C[i][j] en memoria lineal
                result[i * p + j] += a.values[i * n + k] * b.values[k * p + j];
    return move(Tensor({m, p}, result)); // retorna el tensor resultado con su forma correcta
}

// (3.2) MÉTODOS ESTÁTICOS
// Crea tensor lleno de ceros
Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total = 1;
    for (auto s : shape) total *= s; // calcula total multiplicando todas las dimensiones
    return Tensor(shape, vector<double>(total, 0)); // vector<double>(total, 0) crea un vector de 'total' elementos, todos en 0
}

// Crea tensor lleno de unos
Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t total = 1;
    for (auto s : shape) total *= s; // igual que zeros, calcula cuántos elementos necesita
    return Tensor(shape, vector<double>(total, 1));  // vector<double>(total, 1) crea un vector de 'total' elementos, todos en 1
}

// Crea tensor con valores aleatorios en el rango [min, max)
Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    size_t total = 1;
    for (auto s : shape) total *= s;

    vector<double> vals;
    for (size_t i = 0; i < total; i++)
        // rand() genera entero entre 0 y RAND_MAX
        // dividir entre RAND_MAX da un decimal entre 0.0 y 1.0
        // multiplicar por (max-min) y sumar min escala ese decimal al rango [min, max)
        vals.push_back(min + (double)rand() / RAND_MAX * (max - min));
    return Tensor(shape, vals);
}

// Crea tensor 1D con valores secuenciales desde start hasta end (no inclusivo)
Tensor Tensor::arange(int start, int end) {
    vector<double> vals;
    for (int i = start; i < end; i++) vals.push_back(i); 
    return Tensor({(size_t)(end - start)}, vals); // llena con start, start+1, ..., end-1
}

// Seccion(5) TRANSFORMACIONES

// interfaz abstracta = cualquier clase que herede debe implementar apply
class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0; // = 0 significa que es método puramente virtual, obliga a las subclases a implementarlo
    virtual ~TensorTransform() = default;   // destructor virtual necesario para que el polimorfismo libere memoria correctamente
};

// ReLU: reemplaza valores negativos por 0, deja los positivos igual
// Formula: y = max(0, x)
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override { // override indica que estamos implementando el método virtual de la clase padre
        vector<double> v;
        for (size_t i = 0; i < t.total; i++)
            v.push_back(max(0.0, t.values[i])); // si el valor es negativo, lo reemplaza por 0; si es positivo, lo deja igual
        return Tensor(t.shape, v); // retorna nuevo tensor con misma forma pero valores transformados
    }
};

// Sigmoid: comprime los valores al rango (0, 1)
// Fórmula: y = 1 / (1 + e^-x)
class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override { 
        vector<double> v;
        for (size_t i = 0; i < t.total; i++)
            v.push_back(1.0 / (1.0 + exp(-t.values[i]))); // exp(-x) es e^-x, luego aplica la fórmula
        return Tensor(t.shape, v); // retorna nuevo tensor con misma forma pero valores entre 0 y 1
    }
};

// Delegación del apply: el tensor le pasa su propio contenido a la transformación recibida
Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this); // *this es el tensor actual, se lo pasa a ReLU, Sigmoid, etc.
}

// sección 8: concat
// concatena tensores compatibles a lo largo de una dimensión
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty())
        throw invalid_argument("No hay tensores para concatenar");

    vector<size_t> base_shape = tensors[0].shape; // forma del primer tensor, usada como referencia

    if (dim >= base_shape.size())
        throw invalid_argument("Dimension invalida"); // dim debe existir en la forma del tensor

    if (base_shape.size() > 3)
        throw invalid_argument("Maximo 3 dimensiones");

    // valida que todos los tensores tengan el mismo número de dimensiones
    // y que todas las dimensiones (excepto 'dim') sean iguales
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i].shape.size() != base_shape.size())
            throw invalid_argument("Concat incompatible"); // no se puede concat 1D con 2D, por ejemplo

        for (size_t j = 0; j < base_shape.size(); j++) {
            if (j != dim && tensors[i].shape[j] != base_shape[j])
                throw invalid_argument("Concat incompatible"); // las dims que no son 'dim' deben coincidir
        }
    }
    // calcula la forma del tensor resultante:
    // todas las dims igual que base_shape, excepto 'dim' que es la suma de todos
    vector<size_t> new_shape = base_shape;
    new_shape[dim] = 0;
    for (size_t i = 0; i < tensors.size(); i++)
        new_shape[dim] += tensors[i].shape[dim];  // acumula el tamaño de la dimensión concatenada

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
            // concat por filas: se pegan las filas de cada tensor en orden
            for (size_t t = 0; t < tensors.size(); t++) {
                for (size_t i = 0; i < tensors[t].total; i++)
                    result.push_back(tensors[t].values[i]);
            }
        } else {
            // concat por columnas: para cada fila, se toman las columnas de cada tensor
            for (size_t i = 0; i < base_shape[0]; i++) { // recorre filas
                for (size_t t = 0; t < tensors.size(); t++) { // para cada tensor
                    size_t current_cols = tensors[t].shape[1];
                    for (size_t j = 0; j < current_cols; j++) {
                        result.push_back(tensors[t].values[i * current_cols + j]); // i*cols+j convierte 2D a índice lineal
                    }
                }
            }
        }
    }

    // caso 3D
    else if (base_shape.size() == 3) {
        size_t a = base_shape[0]; // dimensión 0 (ej: lotes)
        size_t b = base_shape[1]; // dimensión 1 (ej: filas)

        if (dim == 0) {
            // concat en la primera dimensión: se pegan todos los datos en orden
            for (size_t t = 0; t < tensors.size(); t++) {
                for (size_t i = 0; i < tensors[t].total; i++)
                    result.push_back(tensors[t].values[i]);
            }
        } else if (dim == 1) {
            // concat en la segunda dimensión: para cada lote, se pegan las "filas" de cada tensor
            for (size_t i = 0; i < a; i++) { // recorre dimensión 0
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
            // concat en la tercera dimensión: para cada lote y fila, se pegan las "columnas" de cada tensor
            for (size_t i = 0; i < a; i++) { // recorre dimensión 0
                for (size_t j = 0; j < b; j++) {  // recorre dimensión 1
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

    return Tensor(new_shape, result); // retorna el tensor concatenado con su nueva forma
}

//Seccion 10: Red Neuronal
int main() {
    // Paso 1: Tensor de entrada de dimensiones 1000 x 20 x 20
    Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    cout << "Paso 1 - Entrada:        {1000, 20, 20} -> total: " << input.getTotal() << endl;

    // Paso 2: Aplanar cada muestra de 20x20 a un vector de 400 elementos
    Tensor flat = input.view({1000, 400});
    cout << "Paso 2 - view:           {1000, 400}    -> total: " << flat.getTotal() << endl;

    // Paso 3: Multiplicar por los pesos de la primera capa W1 (400 x 100)
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor z1 = matmul(flat, W1);
    cout << "Paso 3 - matmul W1:      {1000, 100}    -> total: " << z1.getTotal() << endl;

    // Paso 4: Sumar el bias de la primera capa b1 replicado a {1000, 100}
    Tensor b1_row = Tensor::random({1, 100}, 0.0, 0.1);
    vector<double> b1_vals;
    for (size_t i = 0; i < 1000; i++)
        for (size_t j = 0; j < 100; j++)
            b1_vals.push_back(b1_row.getValues()[j]);
    Tensor b1({1000, 100}, b1_vals);
    Tensor a1 = z1 + b1;
    cout << "Paso 4 - suma bias b1:   {1000, 100}    -> total: " << a1.getTotal() << endl;

    // Paso 5: Activación ReLU
    ReLU relu;
    Tensor h1 = a1.apply(relu);
    cout << "Paso 5 - ReLU:           {1000, 100}    -> total: " << h1.getTotal() << endl;

    // Paso 6: Multiplicar por los pesos de la segunda capa W2 (100 x 10)
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor z2 = matmul(h1, W2);
    cout << "Paso 6 - matmul W2:      {1000, 10}     -> total: " << z2.getTotal() << endl;

    // Paso 7: Sumar el bias de la segunda capa b2 replicado a {1000, 10}
    Tensor b2_row = Tensor::random({1, 10}, 0.0, 0.1);
    vector<double> b2_vals;
    for (size_t i = 0; i < 1000; i++)
        for (size_t j = 0; j < 10; j++)
            b2_vals.push_back(b2_row.getValues()[j]);
    Tensor b2({1000, 10}, b2_vals);
    Tensor a2 = z2 + b2;
    cout << "Paso 7 - suma bias b2:   {1000, 10}     -> total: " << a2.getTotal() << endl;

    // Paso 8: Activación Sigmoid
    Sigmoid sigmoid;
    Tensor output = a2.apply(sigmoid);
    cout << "Paso 8 - Sigmoid:        {1000, 10}     -> total: " << output.getTotal() << endl;

    // Verificación: primeras 10 salidas de la muestra 0
    cout << "\nPrimeras 10 salidas de la muestra 0:" << endl;
    for (size_t i = 0; i < 10; i++)
        cout << "  clase[" << i << "] = " << output.getValues()[i] << endl;
    return 0;
}
