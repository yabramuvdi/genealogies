import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from __future__ import unicode_literals, print_function
import plac
import random
import warnings
from pathlib import Path
from spacy.util import minibatch, compounding

all_examples = [
    ("Don Silverio Abondano Prieto, coronel patriota nacido en Santafé en junio de 1799 y fallecido en septiembre de 1835", {'entities': [(4, 28, 'PERSON')]}),
    ("Doña Brigida Gómez de Orozco y Dominguez, esposa en primeras nupcias del Capitán Dionisio de Velasco, alférez mayor y encomendero de San Cristobal", {'entities': [(6, 41, 'PERSON'), (82, 101, 'PERSON')]}),
    ("Don Cristobal de Figueroa y Gómez de Orozco vistió hábitos religiosos", {'entities': [(4, 43, 'PERSON')]}),
    ("El Capitán Pedro de Figueroa y Gómez de Orozco nacido en Pamplona en 1591 y fallecido en 1661", {'entities': [(11, 47, 'PERSON')]}),
    ("Casó en 1757 con don Pedro José de Uribe y Díaz, hijo de don Leonardo de Uribe Salazar y Gómez y de doña Isabel Díaz Mantilla.", {'entities': [(21, 47, 'PERSON'), (61, 94, 'PERSON'), (106, 126, 'PERSON')]}),
    ("Don Jeronimo Gómez de Orozco y Sotomayor nacido en Pamplona en octubre de 1626.", {'entities': [(4, 40, 'PERSON')]}),
    ("Guillermo Acosta Acosta, casado con Ofelia Uribe Duran, de importantes logros políticos, periodista e institutora.", {'entities': [(0, 23, 'PERSON'), (36, 55, 'PERSON')]}),
    ("Doña Alfonsa Ahumada y Bastida, testó en 1804 y murió en 1806: casada con don Cristóbal Antonio del Casal y Freiria.", {'entities': [(6, 31, 'PERSON'), (79, 116, 'PERSON')]}),
    ("Doña Manuela Álvarez y Casal bautizada el 12 de junio de 1740.", {'entities': [(6, 29, 'PERSON')]}),
    ("Doña Juana Francisca de Derrio y Guzmán tuvo de su primer matrimonio con don Lucas Domingo Cortés y Paredes como hija unica a doña Catalina de Cortés y Berrio, esposa del tesorero de la Cruzada don Juan Bautista Herazo y Mendigaña.", {'entities': [(5, 39, 'PERSON'),(77, 108, 'PERSON'), (132, 159, 'PERSON'), (199, 232, 'PERSON')]}),
    ("Don Ramón Argaez y Rodríguez de León nacido en Ansermanuevo el 11 de octubre de 1813.", {'entities': [(4, 36, 'PERSON')]}),
    ("Doña María Eugenia Arjona y Lizarralde, soltera.", {'entities': [(6, 39, 'PERSON')]}),
    ("Desempeñó allí el cargo de alcalde ordinario y fue dueño de la Hacienda.", {'entities': []}),
    ("Doña María Rafaela Perez de Arroyo y Valencia, nacida el 11 de septiembre de 1777.", {'entities': [(6, 46, 'PERSON')]}),
    ("Don Manuel Antonio Arrubla y Martínez, casó con doña Ignada Zuleta y Dominguez (401), hija de don Fernando Zuleta y Córdoba y doña Teresa Dominguez Castillo.", {'entities': [(4, 37, 'PERSON'), (54, 79, 'PERSON'), (99, 124, 'PERSON'), (133, 158, 'PERSON')]}),
    ("Don Jose María Arrubla y Martínez, nació en la ciudad de Antioquia el 4 de mayo de 1780.", {'entities': [(4, 33, 'PERSON')]}),
    ("Había casado el 27 de noviembre de 1839 con doña María Luisa Quevedo y Bemal (295), hija de don José Nicolás Quevedo Castañeda y doña María Teresa Bemal Munar.", {'entities': [(50, 77, 'PERSON'), (97, 128, 'PERSON'), (137, 161, 'PERSON')]}),
    ("El registro civil del matrimonio de don Luis María Azcuenaga Tobar con doña Enriqueta Ricaurte (304), tuvo lugar el 30 de agosto de 1869.", {'entities': [(40, 66, 'PERSON'), (77, 95, 'PERSON')]}),
    ("Don Juan Manuel de Azuero y Arenas, natural del Socorro, nacido aproximadanmente en 1710.", {'entities': [(4, 34, 'PERSON')]}),
    ("Doña Rosalia Azuero Plata, casada con Uberato Gómez Obregón (114).", {'entities': [(6, 26, 'PERSON'), (39, 60, 'PERSON')]}),
    ("Don Juan Rodolfo de Azuero y García, nacido en el Socorro donde casó, el 8 de enero de 1767, con doña Antonia Gertrudis Gómez Farelo y Uribe, bautizada en la misma ciudad, de nueve meses de nacida, el 27 de diciembre de 1743, hija de don Juan Gómez Farelo y Pineda y de su segunda esposa doña Felipa de Uribe Salazar y Lamo.", {'entities': [(4, 35, 'PERSON'), (103, 141, 'PERSON'), (239, 265, 'PERSON'), (295, 325, 'PERSON')]}),
    ("Don Zacarias Azuero-Gómez y Gómez, nacido en el Socorro y fallecido en Bogotá, de donde era vecino hacia tiempo, el 19 de diciembre de 1874.", {'entities': [(4, 33, 'PERSON')]}),
    ("Don José María Baraya y Prieto nació en inmediaciones de Tunja en septiembre 28 de 1825.", {'entities': [(4, 30, 'PERSON')]}),
    ("Don Salvador Ortiz de la Barrera nació en la ciudad de Antequera en 1759.", {'entities': [(4, 32, 'PERSON')]}),
    ("Don Silvestre Ortiz De La Barrera y Silvestre emigró con otros realistas en 1819, y los patriotas le confiscaron sus bienes.", {'entities': [(4, 45, 'PERSON')]}),
    ("Don Gregorio Barreto y Sanabria, nacido en Tunja en 1723 y vecino de Riohacha en 1760.", {'entities': [(4, 31, 'PERSON')]}),
    ("Doña Josefa Gertrudis Barriga y Brito, bautizada en Santafé el 28 de marzo de 1773.", {'entities': [(6, 38, 'PERSON')]}),
    ("Murió en su ciudad natal, colmada de respeto y consideracion, el 31 de octubre de 1851.", {'entities': []}),
    ("Fue casada en primeras nupcias, el 28 de octubre de 1797, con don Juan Esteban de Ricaurte y Mauriz (304), hijo de don Rafael de Ricaurte y Terreros y de doña María Ignacia Mauriz de Posada, y viudo de donna Maria Clemencia lozano y Gonzalez Manrique.", {'entities': [(66, 99, 'PERSON'), (119, 148, 'PERSON'), (160, 190, 'PERSON'), (209, 251, 'PERSON')]}),
    ("Celoso cultivador de la música y acertado interprete.", {'entities': []}),
    ("Gozó de gran aprecio dentro de sus conciudadanos por su amena e interesante conversación llena de recuerdos y anécdotas de la época de la Independencia.", {'entities': []}),
    ("Había casado con doña Justa Martín Nieto y Navarro.", {'entities': [(23, 51, 'PERSON')]}),
    ("Doña Manuela Benito y Tobar, nacida en Zipaquirá en 1813 y fallecida en 1879.", {'entities': [(6, 28, 'PERSON')]}),
    ("El doctor don Enrique Berbeo y Tobar, abogado distinguido, gobernador de la Provincia de Neiva en 1846 y su Prefecto en 1858.", {'entities': [(14, 36, 'PERSON')]}),
    ("Don Miguel Bermudez de Olarte, nacido por los años de 1742, fue vecino importante de Chiquinquirá, donde desempeñó los destinos de administrador de correos, sargento, mayor de milicias y alcalde ordinario de dicha ciudad en varias ocasiones.", {'entities': [(4, 29, 'PERSON')]}),
    ("Doña María Bemal y Herrera nacida en julio de 1658, casada con don Manuel Lopez Basurto.", {'entities': [(6, 27, 'PERSON'), (66, 88, 'PERSON')]}),
    ("Casó en septiembre de 1725 con el capitan Fernando Munar y Arguindegui - (241), hijo del capitan Pedro de Munar y Jurado y de doña Ángela de Arguindegui y López.", {'entities': [(42, 70, 'PERSON'), (97, 120, 'PERSON'), (132, 161, 'PERSON')]}),
    ("Dos hermanos, don Juan Nelson y don Carlos, fueron los fundadores de esta familia en Colombia, oriundos de Kingston, Jamaica, hijos de don Maximiliano Bonitto y de su esposa doña Julia.", {'entities': [(139, 158, 'PERSON')]}),
    ("Doña Ana María de la Borda y Burgos, fallecida en Santafé en mayo de 1749, primera esposa de don Pedro de la Rocha (307), hijo del oidor Domingo de la Rocha Ferrer y de doña Juana Clemencia Labarces y Pando, cartagenera, y quien viudo casó con doña Ignacia de Florez y Subia.", {'entities': [(6, 36, 'PERSON'), (98, 115, 'PERSON'), (138, 164, 'PERSON'), (176, 208, 'PERSON'), (252, 277, 'PERSON')]}),
    ("El Coronel José Cornelio Borda Sarmiento nacido en la hacienda de Turnias, vecindario de Facativá, el 6 de agosto de 1829.", {'entities': [(11, 40, 'PERSON')]}),
    ("Don Juan Borrero y Ramírez, bautizado de un día en la iglesia parroquial de Alosno el 27 de enero de 1739.", {'entities': [(4, 26, 'PERSON')]}),
    ("Bou Martin Boshell Sheppard, nacido en Dublín hacia 1826 y muerto en Bogotá el 30 de junio de 1888.", {'entities': [(0, 27, 'PERSON')]}),
    ("Don Juan Bautista de Brigard y Dombrowski nació en Varsovia, Polonia, tal vez el 24 de junio de 1792, aunque de algunos documentos se deducen otras fechas.", {'entities': [(4, 41, 'PERSON')]}),
    ("Casó en enero de 1858 con donna Ana Joaquina Arenas García.", {'entities': [(32, 58, 'PERSON')]}),
    ("Don Camilo de Brigard Nieto, nacido en septiembre de 1867.", {'entities': [(4, 27, 'PERSON')]}),
    ("Doña Teresa Brush Dominguez, nacida en Nueva York en 1832 y fallecida en Bogotá en mayo de 1904.", {'entities': [(6, 28, 'PERSON')]}),
    ("Don Jacinto de Buenaventura y Lombardo declara en su testamento que es natural de los reinos de España.", {'entities': [(4, 38, 'PERSON')]}),
    ("Don Fernando Jacinto de Buenaventura y Castillo nacido en Ibague en 1754.", {'entities': [(4, 47, 'PERSON')]}),
    ("Dada la epoca, y la coincidencia de los apellidos, pensamos que esta señora podría pertenecer a la familia de don Andres de Bustamante y Caballero.", {'entities': [(115, 147, 'PERSON')]}),
    ("Don Ildefonso José de Buenaventura y Calderón, quien fue fraile franciscano.", {'entities': [(4, 45, 'PERSON')]}),
    ("Casó en Santafé con doña Gregoria de Ricaurte (304), hija legítima del contador don Antonio de Ricaurte y de doña Francisca de Terreros y Santoyo.", {'entities': [(25, 45, 'PERSON'), (84, 103, 'PERSON'), (114, 145, 'PERSON')]}),
    ("Don Rodrigo de Cabrera y Pedraza fue bautizado en la ciudad de Baeza el 27 de abril de 1605, hijo legítimo de don Lorenzo de Cabrera San Martín, bautizado en la misma ciudad el 4 de noviembre de 1570, y de doña Mariana de Pedraza bautizada también allí el 15 de septiembre de 1580, casados el 14 de abril de 1602.", {'entities': [(4, 32, 'PERSON'), (114, 143, 'PERSON'), (211, 229, 'PERSON')]}),
    ("Los solicitantes fueron hijos de don Diego de Pedraza, vecino y corregidor de Baeza y de doña Juana Díaz de Navarrete su esposa, y nietos paternos de don Juan Sánchez de Pedraza, vecino de la ciudad de Sevilla, nacido por los años de 1402.", {'entities': [(37, 53, 'PERSON'), (94, 117, 'PERSON'), (154, 177, 'PERSON')]}),
    ("Don Antonio Gil de Cabrera y Quirós fue en Santafé capitán y Maestre de Campo, Regidor, Alférez Real y Alcalde Ordinario en 171 y 1715, y contador del Juzgado General de Bienes de Difuntos.", {'entities': [(4, 35, 'PERSON')]}),
    ("Nieta paterna de Pedro Gutiérrez y María Sáenz Prieto, y materna de Juan Gómez Portillo, de los compañeros de Quesada y de los fundadores de Santafé, primer encomendero de Usme, natural de Portillo jurisdicción de Toledo, y Catalina Martín Pacheco, oriunda de Carmona en Andalucía.", {'entities': [(17, 32, 'PERSON'), (35, 53, 'PERSON'), (68, 87, 'PERSON'), (224, 247, 'PERSON')]}),
    ("Don Francisco Beltrán de Caicedo fue vecino de la ciudad de Victoria fundada en 1558, de efímera existencia.", {'entities': [(4, 32, 'PERSON')]}),
    ("El licenciado don Fernando de Berrío y Caicedo, casado con doña Guiomar de Berrío y Mendoza, vecina de Granada.", {'entities': [(18, 46, 'PERSON')]}),
    ("Doña Ana María de Fonseca y Olmos, la esposa de López de Mayorga fue bautizada en Santafé el 19 de abril de 1581, hija legítima de Juan de Olmos y de doña Catalina Velásquez.", {'entities': [(5, 33, 'PERSON'), (131, 144, 'PERSON'), (155, 173, 'PERSON')]}),
    ("Don Fernando Leonel de Caicedo y Mayorga nació en Santafé el 27 de junio de 1637 y murió en la misma ciudad el 26 de enero de 1689.", {'entities': [(4, 40, 'PERSON')]}),
    ("Doña Andrea Gertrudis de Caicedo y Solabarrieta, mencionada en el testamento del padre quien solamente la declara a ella y a otro “por nacer”.", {'entities': [(5, 47, 'PERSON')]}),
    ("Don Domingo de Caicedo y Prieto, primogénito, bautizado en Santafé el 24 de septiembre de 1747.", {'entities': [(4, 31, 'PERSON')]}),
    ("Casó en primeras nupcias con don Manuel de Narváez Porras (245), hijo de don Manuel María de Narváez Guerra y de doña Rebeca Porras Barahona.", {'entities': [(33, 57, 'PERSON'), (77, 107, 'PERSON'), (118, 140, 'PERSON')]}),
    ("Don Luis Dionisio Caicedo y Flórez nació en Purificación el 9 de octubre de 1752 y allí fue bautizado, murió el 20 de febrero de 1813.", {'entities': [(4, 34, 'PERSON')]}),
    ("El Teniente Coronel Fernando Caicedo y Sanz de Santamaría nació en Santafé el 13 de mayo de 1796 y murió el 15 de junio de 1864.", {'entities': [(20, 57, 'PERSON')]}),
    ("Colegial del Rosario en 1808, prestó servicios a la independencia a partir del 18 de octubre de 1810, en que se le designó alférez.", {'entities': []}),
    ("Doña Antonia Cajigas y Bernal, esposa de don Clemente Padilla Ahumada (265), nacido en 1802 y fallecido en 1836, hijo de don José Antonio Padilla y Pontón y de doña Manuela Ahumada y Gutiérrez de Lara.", {'entities': [(5, 29, 'PERSON'), (45, 69, 'PERSON'), (125, 154, 'PERSON'), (165, 200, 'PERSON')]}),
    ("Doña Baltasara Caldas y Tenorio, que casó, el 29 de octubre de 1.809, con el doctor Jorge Wallis, médico inglés, tronco de ésta familia en Colombia.", {'entities': [(5, 31, 'PERSON'), (84, 96, 'PERSON')]}),
    ("El doctor don Francisco José de Caldas y Tenorio nació en la jurisdicción de Popayán en fecha que no ha podido precisarse.", {'entities': [(14, 48, 'PERSON')]}),
    ("Doña Francisca Javiera Calderón y Cuadros Rangel, natural de Girón donde casó en 1748 con su primo hermano don Carlos Mantilla de los Ríos y Heras (214), hijo del Capitán Manuel Mantilla de los Ríos y Martín Nieto de Paz, y de doña Angela de las Heras Pantoja y Celis.", {'entities': [(5, 48, 'PERSON'), (111, 146, 'PERSON'), (171, 220, 'PERSON'), (232, 267, 'PERSON')]}),
    ("Muy joven se vinculó al periódico El Espectador y más tarde a El Tiempo.", {'entities': []}),
    ("Casó en esta ciudad el 29 de mayo de 1915 con doña Teresa Nieto Restrepo (247), hija de don Miguel Nieto Ricaurte y de doña María de Jesús Restrepo Santa María.", {'entities': [(51, 72, 'PERSON'), (92, 113, 'PERSON'), (124, 159, 'PERSON')]}),
    ("Don Nicolás Ignacio Calvo y Vargas casó con don María Manuela de la Serna Mujica y Olarte, hija legítima de don Nicolás Antonio de la Serna Mujica y Vergara y de doña María Teresa de Olarte y Herrera.", {'entities': [(4, 34, 'PERSON'), (48, 89, 'PERSON'), (112, 156, 'PERSON'), (167, 199, 'PERSON')]}),
    ("Don José María Candelario Camacho y Hernández, bautizado en la parroquia de Santa Bárbara en Santafé el 4 de febrero de 1776.", {'entities': [(4, 45, 'PERSON')]}),
    ("Don Fermín Camacho y Quevedo, nacido en 1812 y muerto en 1879.", {'entities': [(4, 28, 'PERSON')]}),
    ("Dona María Josefa Camacho y Quevedo, muerta en mayo de 1860, esposa de don Francisco Quijano Caicedo (296), hijo de don José María Quijano y Vánegas y doña María Josefa Caicedo y Santamaría.", {'entities': [(5, 35, 'PERSON'), (75, 100, 'PERSON'), (120, 148, 'PERSON'), (156, 189, 'PERSON')]}),
    ("Don Luis Camacho de la Peña y Guzmán nació en 1653, y fue en Tunja capitán de caballos corazas, regidor, encomendero, alcalde ordinario entre los años de 1686 a 1717, y se le llamaba también Luis Camacho de Guzmán.", {'entities': [(4, 36, 'PERSON'), (191, 213, 'PERSON')]}),
    ("Don José Manuel Camacho y Lozano fue hijo de don Manuel Antonio Camacho y Moya y de doña Saturia Lozano Vilianueva.", {'entities': [(4, 32, 'PERSON'), (49, 78, 'PERSON'), (89, 114, 'PERSON')]}),
    ("El coronel don José María Cancino y Madero, prócer de la Independencia, nacido en Santafé por los años de 1790.", {'entities': [(15, 42, 'PERSON')]}),
    ("El doctor don Manuel Antonio del Cantillo y Fernández, recibido de abogado en 1825, se distinguió en su profesión.", {'entities': [(14, 53, 'PERSON')]}),
    ("Doña María de Jesús Cantillo Borda, nacida en Bogotá en 1845, casada con don Carlos Carrasquilla y Lema (78), hijo de don Juan Manuel Carrasquilla y Posada y de doña Candelaria Lema y Álvarez.", {'entities': [(5, 34, 'PERSON'), (77, 103, 'PERSON'), (122, 155, 'PERSON'), (166, 191, 'PERSON')]}),
    ("Don José Carbonell nació en España en 1737, y falleció en Santafé donde fue sepultado el 24 de noviembre de 1782; hijo legítimo de don Rafael Carbonell y de doña María Rojas y Santa Cruz.", {'entities': [(4, 18, 'PERSON'), (135, 151, 'PERSON'), (162, 186, 'PERSON')]}),
    ("Casó a los trece años de edad, en 1786 con don Rafael Araos Ricaurte, nacido en 1737, hijo de don Manuel García de Araos y de doña Rosalía de Ricaurte y Terreros, y viudo de doña Josefa Tobar y Andrade.", {'entities': [(47, 68, 'PERSON'), (98, 120, 'PERSON'), (131, 161, 'PERSON'), (179, 201, 'PERSON')]}),
    ("Don Francisco Antonio Copete y Almansa fue bautizado en Santafé el 29 de enero de 1792.", {'entities': [(4, 38, 'PERSON')]}),
    ("Don Joaquín Fernández de Córdoba y Valencia nació en Popayán el 6 de noviembre de 1750, y vistió la beca de San Bartolomé en 1765.", {'entities': [(4, 43, 'PERSON')]}),
    ("Don José Antonio Fernández de Córdoba y Navarro, natural de Santafé quien casó el 31 de marzo de 1846 en Cuba con doña María del Pilar Valdés, natural de La Habana, ciudad donde se radicaron.", {'entities': [(4, 47, 'PERSON'), (119, 141, 'PERSON')]}),
    ("Don Isidoro Cordovez y Caso nació en La Serena en 1797 y murió en Bogotá de más de cincuenta años.", {'entities': [(4, 27, 'PERSON')]}),
    ("Don Fermín Camacho y Quevedo, nacido en 1812 y muerto en 1879.", {'entities': [(4, 28, 'PERSON')]}),
    ("Don Higinio Cualla y Caicedo nació en Santafé, abrazó la carrera militar y vivió por mucho tiempo en Cartagena.", {'entities': [(4, 28, 'PERSON')]}),
    ("Don Juan de Cuéllar y Sotelo natural de La Palma, hoy Cundinamarca, fue depositario general de dicha ciudad y casó por los años de 1610 a 1615 con doña Bernarda de Herrera Montalvo y Reyes, oriunda del mismo lugar, hija de don Lázaro de Herrera Montalvo y de doña Juana de los Reyes, peninsulares, de quienes sólo sabemos que ‘fueron personas que sirvieron a Su Majestad”.", {'entities': [(4, 28, 'PERSON'), (152, 188, 'PERSON'), (227, 253, 'PERSON'), (264, 282, 'PERSON')]}),
    ("Don Tomás de Cuenca y Castillo nacido en Neiva hacia 1710 y muerto en 1774, era hijo de don Juan de Cuenca Moscoso y de doña Lucía Potenciaría del Castillo Perdomo.", {'entities': [(4, 30, 'PERSON'), (92, 114, 'PERSON'), (125, 163, 'PERSON')]}),
    ("Álvaro Chacón de Luna y Gorraiz, fue bautizado en Vélez el 4 de diciembre de 1640 y sepultado allí el 10 de julio de 1698.", {'entities': [(0, 31, 'PERSON')]}),
    ("Don José Ignacio de la Torre Forero, bautizado en Suesca el 10 de febrero de 1758.", {'entities': [(4, 35, 'PERSON')]}),
    ("Casó en Tenjo en 1783, con doña Rita Luque García, hija de Jorge Luque y de doña Bárbara García y González.", {'entities': [(32, 49, 'PERSON'), (59, 70, 'PERSON'), (81, 106, 'PERSON')]}),
    ("Casó en el citado pueblo el 14 de octubre de mismo año de 1807, con doña María del Pilar Araos y Sánchez Borda (17), hija de don Antonio Araos Ricaurte y de doña Inés Sánchez Borda.", {'entities': [(73, 110, 'PERSON'), (129, 151, 'PERSON'), (162, 180, 'PERSON')]}),
    ("Don José Antonio de la Torre Araos nació en Suesca el 9 de octubre de 1808, y murió vilmente asesinado.", {'entities': [(4, 34, 'PERSON')]}),
    ("Don Alejo de la Torre Araos nació en Chocontá el 20 de julio de 1812 y falleció en Bogotá el 3 de septiembre de 1864, hijo de don José Ramón de la Torre Prieto y de María del Pilar Araos y Sánchez Borda.", {'entities': [(4, 27, 'PERSON'), (130, 159, 'PERSON'), (165, 202, 'PERSON')]}),
    ("Don Simón de la Torre Herrera, fallecido en agosto de 1937, hijo de don Ignacio de la Torre Araos y de doña Teresa de Herrera Imbrechts.", {'entities': [(4, 29, 'PERSON'), (72, 97, 'PERSON'), (108, 135, 'PERSON')]}),
    ("Fue hijo de don Juan Crisóstomo Dimas de la Torre Navarrete y de doña Beatriz Forero Rubiano, nacida en 1725.", {'entities': [(16, 59, 'PERSON'), (70, 92, 'PERSON')]}),
    ("Don Elíseo de la Torre González murió en Apulo en abril de 1918, casado dos veces, la primera con su prima doña Isabel Villar de la Torre, mencionada, hija de don Valentín Villar y de doña Natalia de la Torre Córdoba; y la segunda, con otra prima suya, doña Josefina de la Torre Pinzón, también mencionada, hija de don Miguel de la Torre Córdoba y doña Josefa Pinzón Bustamante.", {'entities': [(4, 31, 'PERSON'), (112, 137, 'PERSON'), (163, 178, 'PERSON'), (189, 216, 'PERSON'), (258, 285, 'PERSON'), (319, 345, 'PERSON'), (353, 377, 'PERSON')]}),
    ("Don Matías de Francisco Martín nació en el lugar de Salduero el 16 de marzo de 1767; murió en Bogotá el 26 de noviembre de 1816.", {'entities': [(4, 30, 'PERSON')]}),
    ("Don Juan José D’Elhúyar nació en la ciudad de Logroño, España, y murió en septiembre de 1796 en Santafé.", {'entities': [(4, 23, 'PERSON')]}),
    ("Don José Antonio Salgar nació en el lugar de Sueco, Obispado de Burgos, hijo legítimo de don Felipe Salgar y La Torre y doña Ana Pedriza y Colina", {'entities': [(4, 23, 'PERSON'), (93, 117, 'PERSON'), (125, 145, 'PERSON')]}),
    ("Doña Fructuosa Salgar y González de Noriega, casada en 1799 con don Marcos Gutiérrez Lazo (163 bis), hijo legítimo de don Francisco Javier Gutiérrez Martínez, español, alcalde de Girón, y de doña Fiomiliana Duarte Cornejo.", {'entities': [(5, 43, 'PERSON'), (68, 89, 'PERSON'), (122, 157, 'PERSON'), (196, 221, 'PERSON')]})
]


train_data, test_data = train_test_split(all_examples, test_size=0.15)

# Code from: https://spacy.io/usage/training
def train_model(train_data, model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("es")  # create blank Language class
        print("Created blank 'es' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if
                   pipe not in pipe_exceptions]

    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    #     # test the trained model
    #     for text, _ in TRAIN_DATA:
    #         doc = nlp(text)
    #         print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #         print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model on the training data
        print("Loading from", output_dir)
        nlp = spacy.load(output_dir)
        for text, _ in train_data:
            doc = nlp(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# execute NER training
train_model(train_data, output_dir='/models', n_iter=50)

# test the saved model
nlp = spacy.load('/models')
for text, _ in test_data:
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])