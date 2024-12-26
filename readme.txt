Git bisect es una herramienta para encontrar el commit que introdujo un error.

Inicias el proceso con git bisect start
Marcas el estado actual como malo (git bisect bad)
Indicas un commit anterior donde todo funcionaba (git bisect good)
Git hace checkout automático a commits intermedios
Vas marcando cada commit como bueno o malo según corresponda
Git reduce el rango de búsqueda hasta encontrar el commit exacto
Terminas con git bisect reset

