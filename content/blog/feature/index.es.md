+++
title = "Una breve y práctica introducción a la visualización de features en PyTorch"
date = 2024-09-27

[taxonomies]
tags = ["interpretability", "ml"]
+++

> [**This post is available in English.**](/blog/feature)

{{ video(path="/blog/feature/castle_sped.mp4", caption="Visualización de 'castillo' (483)", autoplay=true) }}

A medida que la inteligencia artifical adquiere mayor capacidad, es adoptada por ambos la industria y el gobierno como una tecnología estándar.
La IA permite la automatización del trabajo del conocimiento. Con suficientes datos, poder computacional y mejoras algorítmicas, una IA podrá ocupar
puestos investigativos, administrativos o de ingeniería.

Dependiendo de cuánto estos nos ayuden a alcanzar nuestros objetivos, puede ser a beneficio o detrimento de la humanidad.
Actualmente hay muchos problemas obstruyendo el avance, entre ellos:

1. No entendemos cómo *realmente* funcionan los modelos de deep learning.
2. No sabemos cómo hacer que los modelos de deep learning hagan *precisamente* lo que queremos.
3. No sabemos *qué* queremos. (O mejor dicho, definir precisa y explícitamente qué queremos)
4. No sabemos qué regulaciones son efectivas para reducir el mal uso de la IA.
5. *Muchos más...*

Mientras que estos problemas son clave, este artículo se enfoca en el problema 1: **interpretabilidad**. No soy un experto en el campo, pero entiendo *justo lo suficiente* para dar una diminuta mano a aquellos que deseen aprender más del tema.

<!-- TODO: You can check out these resources to answer the other questions... -->

> **Aviso**: Asumo un poco de entendimiento de [redes neuronales feed-forward (MLP)](https://www.youtube.com/watch?v=aircAruvnKk) y [gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w).

## ¿Qué resuelve la interpretabilidad?

Volvamos al enunciado del problema.

> No entendemos cómo *realmente* funcionan los modelos de deep learning.

A qué me refiero por "¿cómo realmente funcionan?"

Por decadas, hemos construído diferentes arquitecturas de deep learning, como multi-layer perceptrons; redes neuronales convolucionales, residuales y recurrentes; LSTMs; transformers; y muchos más. Durante el entrenamiento, estas máquinas codifican patrones y algoritmos dentro de sus parámetros (weights, biases, etc). Entendemos cómo funciona el proceso, pero no específicamente los patrones y algorítmos que emergen.

Una vez entrenado, **los modelos son cajas negras**: mientras les alimentemos inputs y recibamos outputs correctos, no entendemos como el modelo logró eso a un nivel neuronal. Sabemos que las neuronas están conectadas para eso, pero ¿*cómo*?

Presentaré un ejemplo para mostrar a qué me refiero. Considerá a [este gato](https://unsplash.com/photos/black-and-white-cat-lying-on-brown-bamboo-chair-inside-room-gKXKBY-C-Dk).

{{ img(path="/blog/feature/cat.jpg", caption="[*(Dale. Considerale.)*](https://aisafety.dance/)") }} 

Si alimentamos esta imagen a un modelo de clasificación como ResNet18, sería clasificado como "Gato egipcio", lo cual no estaría lejos de la verdad.
No existe una categoría "gato" entre los posibles outputs del modelo, así que sería imposible que el modelo sencillamente responda "gato." Practicamente, está correcto!

<figure class="extended-figure">
    <img src="/blog/feature/ResNet18.excalidraw.svg" />
</figure>

¿Pero cómo llegó el modelo a esa conclusión? Hagamos un análisis en reversa.

- Ya que la clase de ImageNet "Gato egipcio" tiene un índice 285, sabemos que en la última capa fully-connected del modelo (`fc`), la neurona 285 tiene la mayor activations de las neuronas de aquella capa (que son 1000 en total). Esto es por la última operacion (`argmax`) devuelve el índice de la neurona de la capa anterior con la mayor activación.
- La neurona 285 de `fc` fue activada por algunas activaciones neuronales de la capa anterior (`avgpool`).
- Neuronas de `avgpool` que contribuyeron a la neurona 285 de `fc` vienen del output de una capa convolucional.
- Esta capa convolucional fue calculada con los resultados de otra capa convolucional.
- Y así sucesívamente hasta que lleguemos a la primera capa convolucional, que está directamente conectada a nuestra imagen (gatito.)

De este análisis surgen muchas preguntas:

1. **Identificación de circuitos:** ¿qué neuronas en `avgpool`, una vez activadas, activan por consecuente a la neurona 285 de la capa `fc`? ¿Y en la capa anterior? ¿Qué circuito complejo de neuronas se ha formado a través de las capas de la red para que el modelo llegue a la conclusión de que esta imagen corresponde a un "Gato egipcio?"
2. **Visualización:** ¿qué representan los disparos de estas neuronas? ¿Corresponden a conceptos, formas u objetos? ¿Podemos ver una imagen de qué representa una sola neurona? ¿Qué hay acerca de un conjunto de neuronas?
3. **Atribución:** ¿qué partes de la imagen original contribuyeron a que el modelo concluya "Gato egipcio?" ¿Qué partes no? ¿Qué partes de la imágen contribuyeron a una neurona particular a disparar? ¿Qué imágenes hacen que una neurona dispare?

Hablando superficialmente, interpretabilidad intenta responder estas preguntas.

Hoy, vamos a intentar responder algunas preguntas sobre *visualización*.

## Definiendo visualización

En el caso general de una red neuronal, podemos definir *visualización de features* como generar un input que maximice la activación de una parte de la red: una neurona de output, una neurona de un hidden-layer, un conjunto de neuronas o una capa entera.

En el caso de un modelo de clasificación de imágenes, con *visualización* de features literalmente nos referimos a generar una imagen. Digamos que hacemos visualización de *clases*, donde optimizamos una imagen como para que el modelo lo clasifique dentro de una clase particular (significando que la neurona correspondiente a esa clase en la última capa será significativamente activada, más que todas las otras neuronas de output.)

In a perfect world, if we were to visualize class 285 on ResNet18, we would get an image of a cute kitten. In reality, though, feature visualizations can be confusing and unintelligible compared to a natural picture. We'll see this as we try to implement it ourselves.

En un mundo perfecto, si fueramos a visualizar la clase 285 en ResNet18 obtendríamos una imagen de un gatito. Sin embargo, las visualizaciones de features en realidad tienden a ser confusas e ininteligibles comparadas a una foto natural. Veremos esto a medida que lo implementemos nosotros mismos.

## Implementando visualización

Utilizando PyTorch, implementaremos visualización de clases para un modelo de clasificación preentrenado. Vamos a optimizar una imagen para que el modelo la clasifique en una clase de ImageNet de nuestra elección. Entonces, ¿qué clase elegiremos?

{{ img(path="/blog/feature/hen.jpg", caption="Clase 8 de ImageNet: *gallina*. [Source.](https://unsplash.com/photos/brown-and-red-he-n-G61iAuzI9NQ)") }} 

¿Por qué gallinas? Porque **todas** tienen carúnculas rojas que son fácilmente reconocibles. Entonces, será más fácil saber si nuestra visualización funciona o no desde ya. (Si no vemos formas rojas, algo anda mal.)

> **En caso que quieras usar otra clase de ImageNet**, [aquí está la lista de la cual puedes elegir](https://github.com/pytorch/hub/blob/c7895df70c7767403e36f82786d6b611b7984557/imagenet_classes.txt). Una vez que te decidiste, grabá el número de línea del mismo y restale 1 para obtener el índice de la clase. (Esto es porque los números de línea empiezan en 1, mientras que los tensores de PyTorch empiezan en 0)

Vamos a visualizar features del modelo ResNet18. Obtuve resultados decentes con este modelo durante mi propia experimentación. Es posible obtener mejores visualizaciones con modelos más grandes como VGG19, pero al costo de la velocidad de la optimización. En este caso, prefiero feedback loops más rápidos a mejor calidad, porque estas nos permitirán experimentar con facilidad.

### Caso base: Optimizar el input como optimizar parámetros de un modelo

Empezaremos el código importando matplotlib como nuestro backend para mostrar imágenes convenientemente. También definiremos una función `ptimg_to_mplimg` para convertir imágenes de un tensor de PyTorch a un array de numpy, disponibilizandolo así para visualización en matplotlib. Definiremos `show_img` para visualizar imágenes concisamente en una sola llamada de función.

```python
import torch
import torchvision
import matplotlib.pyplot as plt

def ptimg_to_mplimg(input: torch.Tensor):
    with torch.no_grad():
        return input.detach().squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
def show_img(input: torch.Tensor, title: str):
    plt.imshow(ptimg_to_mplimg(input))
    plt.title(title)
    # Setting `block=False` and calling `plt.pause` allow us to display the progress in real time
    plt.show(block=False)
    plt.pause(0.1)
```

Con el boilerplate ya escrito, vamos a implementar la forma más simple y obvia de hacer visualización de clases. Vamos a guiarnos de como solemos optimizar redes neuronales: usando un optimizador built-in de PyTorch que ajusta parámetros para minimizar nuestro loss function (función de pérdida). Aquí, intentaremos lo mismo, pero en vez de optimizar los parámetros de un modelo, optimizaremos el input (nuestra imagen). "¿Cuál será nuestro loss function?", te estarás preguntando. Responderemos aquello después.

Descargemos el modelo preentrenado.

```python
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.IMAGENET1K_V1")
# Ponemos el modelo en modo de evaluación
# Desactiva capas dropout y de batch normalization, las cuales necesitamos ahora
model.eval()
```

Necesitamos definir la imagen que será nuestro punto de inicio para la optimización. Usaremos valores random de 0 a 1. Es importante notar que podemos iniciar con cualquier imagen.

```python
input = torch.rand(1, 3, 299, 299, requires_grad=True)
```

Declaramos nuestra clase inicial como "gallina".

```python
TARGET_CLASS = 8 # kokoroko!
```

Hmm, ¿qué optimizador deberíamos usar? ¿Por qué no el clásico SGD? (stochastic gradient descent)

```python
LR = 0.5
optimizer = torch.optim.SGD([input], lr=LR)
```

Crearemos una función que realiza un paso de optimización para que nuestro código esté "ordenado".

```python
def step():
    optimizer.zero_grad()
    output = model(input)
    loss = -output[:, TARGET_CLASS].mean()
    loss.backward()
    optimizer.step()
```

Acá hicimos algo importante: **definimos nuestro loss function como el negativo de la activación de la neurona de output correspondiente a nuestra clase target.** Queremoos que la activación de esta neurona sea lo mayor posible. Nuestro optimizador intenta *minimizar* el loss function, por lo tanto, si definimos el loss function como el negativo de la activación de nuestra neurona target, el optimizador intentará maximizar la activación de la neurona target.

Ahora, sencillamente necesitamos llamar la función "step" en un bucle, mostrando nuestra imagen cada tanto para ver el progreso.

```python
STEPS = 200
for i in range(STEPS):
    step()

    # Mostrar la imagen cada 10 pasos
    if i % 10 == 0:
        print(f"Step {i}/{STEPS}")
        show_img(input, f"Visualization of class {TARGET_CLASS}")
```

Okay! Completamos nuestra primera versión. Veamos qué tal le va.

{{ video(path="/blog/feature/app0.mp4") }}

Hmm, no parece tanto una gallina. ¿Qué podemos hacer para resolver esto?

### Mejora 1: Cambiar la imagen inicial

Iniciar con valores random de 0 a 1 puede estar causando que el optimizador tienda a valores extremos (fuera del rango RGB 0-1). No solo esto genera alto contraste en la imagen, pero si queremos deshacernos del ruido, iniciar con una imagen ya ruidosa capaz no sea la mejor opción. Vamos a intentar una imágen más uniformemente gris. Siendo preciso, usarémos exactamente el mismo ruido pero con una media de 0.5 y un rango de 0.49-0.51.

Vamos a reflejar estos cambios en el código.

```python
input = (torch.rand(1, 3, 299, 299) - 0.5) * 0.01 + 0.5
input.requires_grad = True
```

{{ video(path="/blog/feature/app1.mp4") }}

Mucho mejor! Si uno entrecierra los ojos, las cabezas rojas de las gallinas resaltan, mientras que en el resto de la imagen, emergen patrones similares a plumas.

### Mejora 2: Robustez transformacional

Hemos estado calculando los gradientes basado en la misma imagen, a la misma escala, rotación y translación para cada paso.
Esto significa que nuestro código optimiza para que la imagen sea clasificada como "gallina" solo desde *un* punto de vista.
Si rotamos la imagen, nada nos asegura que seguirá siendo clasificada como "gallina".
Nuestra imagen no es *transformacionalmente robusta*.

Si queremos que el modelo reconozca nuestro target class en una imagen despues de ser escalada o rotada, debemos incluir aquel proposito en nuestra optimización.

> **Por qué esto ayudaría con el problema del ruido?**

No me está tan claro el porqué, sinceramente.
Hablando *vagamente*, introducir transformaciones estocásticas parece prevenir que el optimizador se mantenga pegado a algún patron ruidoso.
Para una explicación mucho mejor, podés revisar [la sección de *Transformational robustness* del papel de *Feature Visualization* en Distill.](https://distill.pub/2017/feature-visualization)

Anyways, the implementation involves applying random transformations on the input before the optimization.

Before our step function, we must define what transforms we'll apply to our image before passing it to our model.

```python
from torchvision import transforms as T
transforms = T.Compose([
    # Applies minor changes in brightness, contrast, saturation, and hue
    T.ColorJitter(brightness=(0.98, 1.02), contrast=(0.99, 1.01), saturation=(0.98, 1.02), hue=(-0.01, 0.01)),
    # Rotates the image by 15deg clockwise or counter-clockwise. We also apply a bit of random zoom in and out.
    T.RandomAffine(degrees=(-15.0, 15.0), scale=(0.96,1.04), fill=0.5),
])
```

On our step function, we must modify the forward-pass call.

```python
def step():
    optimizer.zero_grad()
    output = model(transforms(input))	# <-- this line right here
    loss = -output[:, TARGET_CLASS].mean()
    loss.backward()
    optimizer.step()
```

{{ video(path="/blog/feature/app2.mp4", caption="Result after optimizing for transformational robustness.") }}

> **Note on gradient propagation:** While I was originally implementing transformational robustness, I misunderstood its concept and *actually transformed the visualization*, instead of *just doing gradient propagation on* the transformed image. The difference is in the step function:
> ```python
> # Case 1: ACTUALLY TRANSFORMING THE VISUALIZATION (don't try at home)
> input = transforms(input.detach())
> output = model(input)
> # Case 2: JUST DOING GRADIENT PROPAGATION ON THE TRANSFORMED IMAGE
> output = model(transforms(input))
> ```

### Improvement 3: Implement L2 regularization

L2 regularization. Hmm, what? It's simply adding the square of the parameters we're optimizing to our loss function. Actually, we add the sum of the squares of our parameters multiplied by a coefficient, usually called λ (lambda). In this case, the parameters are the color values for each pixel.

This basically penalizes color values that deviate significantly from 0. In our case, we want values that stray too far from 0.5, the "middle" point between 0.0 and 1.0, the range of color values. This allows our optimization to have a balance between maximizing our target activation (be it class, neuron, layer, whatever) and having our image be in a valid color range. This will get rid of values that are too extreme on the red, green, or blue channels.

Implementing it is quite easy. We only need to define λ and change a line in the loss function definition.

```python
LR = 1.5 # We'll also increase the learning rate.
L2_lambda = 0.05
```
In our step function:

```python
def step():
    optimizer.zero_grad()
    output = model(transforms(input))
    loss = -output[:, TARGET_CLASS].mean() + L2_lambda * ((input - 0.5) ** 2).sum()     # <-- this line
    loss.backward()
    optimizer.step()
```

{{ video(path="/blog/feature/app3.mp4") }}

### Improvement 4: Blur the image every few steps

Now, for a final technique, I'll introduce a somewhat obvious technique to get rid of noise, which can work surprisingly well: simply applying Gaussian blur to the image. Of course, if we do this repeatedly, we will only get a blurry image. If done occasionally, though, we can obtain good results.

We will add a parameter to our step function so we can know our step index.
Arbitrarily, I've set the step function to blur the image every 10 steps.

```python
def step(i: int):
    optimizer.zero_grad()
    output = model(transforms(input))
    loss = -output[:, TARGET_CLASS].mean() + L2_lambda * ((input - 0.5) ** 2).sum()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        with torch.no_grad():
            input.copy_((T.GaussianBlur(5, 1.5))(input))


STEPS = 200
for i in range(STEPS):
    step(i)
    # ... the rest of the code
```

This will coincide exactly with our image display, so you can see the blur effect.
To prevent this coincidence you can change the blurring condition to `i % 10 == 1`.
This will make the blurring occur exactly after displaying the image, instead of before.

{{ video(path="/blog/feature/app4.mp4", caption="") }}
 

## Testing our feature visualization on various classes

Now that we've got something working, let's try our model with lots of different classes:

{{ video(
path="/blog/feature/conclusion.mp4",
caption="From top-left to bottom-right: hen (8), pelican (144), leopard (288), hammer (587), iPod (605), slot (800), potpie (964), scuba diver (983).",
extended=true
) }}

### Weird experimentation (eternal zoom-in)

While initially implementing transformational robustness, I misunderstood the concept and *actually transformed the visualization*, instead of *just doing gradient propagation on* the transformed image. During these trying times, I experimented with biasing the transformations to continually increase the scale of the image. The result is an eternal zoom-in effect, but the atmosphere is that of endlessly submerging yourself in alien worlds.

{{ video(path="/blog/feature/alien3.mp4", caption="Left: 'hen' (8). Right: 'robin' (15)", extended=true) }}

I also implemented a sort-of L1 regularization where simply the color values multiplied by a lambda coefficient are added to the loss function. This gives some very trippy results.

{{ video(path="/blog/feature/alien.mp4", caption="Left: 'hen' (8). Right: 'robin' (15)", extended=true) }}

I'd dare say that these moving visualizations give us a different perspective to understand what a feature represents.

{{ video(path="/blog/feature/barbellwine.mp4", caption="Left: Barbell World. Right: Wine World.", extended=true) }}

## Limitations

There are definitely some limitations to our current approach:

- L2 regularization inherently reduces color contrast since we unfavor color values that stray far from [0.5, 0.5, 0.5].
- ResNet18 is a relatively small model, so its visualizations may not be as high quality as those from larger models.
- ResNet18 seems to have a bias towards the color green. This is a common feature among many of the models I've tested. *My theory*: since many of the classes and thus training data are animal-related, grass is a common denominator in the background of these pictures. Therefore, the model sees a lot of green during its training, which generates a bias.
- [There are better alternatives to Gaussian blur for our purpose.](https://en.wikipedia.org/wiki/Bilateral_filter)

Reading [this article will give you an idea of how good feature visualization can look when done right](https://distill.pub/2017/feature-visualization/).

## Conclusion

That's good enough for the scope of this post. I hope you found interpretability to be fun and interesting. 

And hey, if you did, don't stop here! I barely scratched the surface of what the field actually revolves around. There are a bunch of resources for you to continue researching:

- [Distill's Circuits Thread](https://distill.pub/2020/circuits/): Great and accessible interpretability papers. Contains [a paper specifically related to feature visualizations](https://distill.pub/2017/feature-visualization).
- [Transformer Circuits Thread](https://transformer-circuits.pub/): State-of-the-art interpretability papers about transformers and language models.
- [ARENA Course](https://www.arena.education/): Hands-on practice for AI alignment/interpretability skills.
