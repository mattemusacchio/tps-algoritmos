#include "tp3.h"
#include <stdlib.h>
#include <string.h>

struct dictionary {
  size_t size;
  size_t capacity;
  destroy_f destroy;
  struct entry_node **buckets;
};

struct entry_node {
  char *key;
  void *value;
  struct entry_node *next;
};

typedef struct entry_node entry_node_t;

#define UMBRAL 0.7

void entry_node_destroy(struct entry_node *node, destroy_f destroy) {
  if (destroy) destroy(node->value);
  free(node->key);
  free(node);
};

// hash de FNV-1a
unsigned long hash(const char *key, size_t capacity) {
    unsigned long hash = 14695981039346656037UL;
    const unsigned char *ptr = (const unsigned char *)key;

    while (*ptr) {
        hash ^= *ptr++;
        hash *= 1099511628211UL;
    }

    return (unsigned long) (hash % capacity);
}


void rehash(dictionary_t *dictionary) {
    size_t new_capacity = dictionary->capacity * 2;
    struct entry_node **new_buckets = calloc(new_capacity, sizeof(struct entry_node *));
    if (!new_buckets) return;
    for (size_t i = 0; i < dictionary->capacity; i++) {
        struct entry_node *node = dictionary->buckets[i];
        while (node) {
        struct entry_node *next = node->next;
        unsigned long index = hash(node->key, new_capacity);
        node->next = new_buckets[index];
        new_buckets[index] = node;
        node = next;
        }
    }
    free(dictionary->buckets);
    dictionary->buckets = new_buckets;
    dictionary->capacity = new_capacity;
};

dictionary_t *dictionary_create(destroy_f destroy) {
  dictionary_t *dictionary = malloc(sizeof(dictionary_t));
  if (!dictionary) return NULL;
  dictionary->size = 0;
  dictionary->capacity = 100;
  dictionary->destroy = destroy;
  dictionary->buckets = calloc(dictionary->capacity, sizeof(struct entry_node *));
  if (!dictionary->buckets) {
    free(dictionary);
    return NULL;
  }
  return dictionary;
};

bool dictionary_put(dictionary_t *dictionary, const char *key, void *value) {
/* Inserta un par clave-valor en el diccionario. O(1).
 * Pre-condiciones:
 * - El diccionario existe
 * - La clave tiene largo mayor a cero
 * - El valor puede ser destruido con la función con la que se inicializó el diccionario.
 * Post-condiciones:
 * - Retorna true si se ha podido guardar con éxito el par clave/valor
 * - Retorna false de otro modo.
 * - Si la clave ya estaba presente, se elimina el valor previo
*/

    if ((float) dictionary->size / (float) dictionary->capacity >= UMBRAL) {
        rehash(dictionary);
    }
    unsigned long index = hash(key, dictionary->capacity);
    entry_node_t *node = dictionary->buckets[index];
    while (node) {
        if (strcmp(node->key, key) == 0) {
          if(dictionary->destroy) dictionary->destroy(node->value);
            node->value = value;
            return true;
        }
        node = node->next;
    }

    entry_node_t *new_node = malloc(sizeof(entry_node_t));
    if (!new_node)
        return false;

    new_node->key = malloc(strlen(key) + 1);
    if (!new_node->key) {
        free(new_node);
        return false;
    }
    strcpy(new_node->key, key);
    new_node->value = value;
    new_node->next = dictionary->buckets[index];
    dictionary->buckets[index] = new_node;
    dictionary->size++;

    return true;
};

void *dictionary_get(dictionary_t *dictionary, const char *key, bool *err) {
  /* Obtiene un valor del diccionario desde su clave. O(1).
 * Pre-condiciones
 * - El diccionario existe
 * - La clave tiene largo mayor a cero
 * Post-condiciones:
 * - Si la clave está presente, retorna el valor asociado y err debe ser false
 * - De otro modo, debe retornar NULL y err debe ser true
 */
  unsigned long index = hash(key, dictionary->capacity);
  struct entry_node *node = dictionary->buckets[index];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      *err = false;
      return node->value;
    }
    node = node->next;
  }
  *err = true;
  return NULL;
};

bool dictionary_delete(dictionary_t *dictionary, const char *key) {
  /* Elimina una clave del diccionario. O(1).
 * Pre-condiciones
 * - El diccionario existe
 * - La clave tiene largo mayor a cero
 * Retorna true si la clave estaba presente y se pudo eliminar, o false
 * de otro modo.
 */
  unsigned long index = hash(key, dictionary->capacity);
  struct entry_node *node = dictionary->buckets[index];
  struct entry_node *prev = NULL;
  while (node) {
    if (strcmp(node->key, key) == 0) {
      if (prev) {
        prev->next = node->next;
      } else {
        dictionary->buckets[index] = node->next;
      }
      entry_node_destroy(node, dictionary->destroy);
      dictionary->size--;
      return true;
    }
    prev = node;
    node = node->next;
  }
  return false;
};

void *dictionary_pop(dictionary_t *dictionary, const char *key, bool *err) {
  /* Elimina una clave y retorna su valor asociado. O(1).
 * Pre-condiciones:
 * - El diccionario existe
 * - La clave tiene largo mayor a cero
 * Post-condiciones:
 * - Si la calve está presente, retorna el valor asocaido y err debe ser false
 * - De otro modo, debe retornar NULL y err debe ser true
 */
  unsigned long index = hash(key, dictionary->capacity);
  struct entry_node *node = dictionary->buckets[index];
  struct entry_node *prev = NULL;
  while (node) {
    if (strcmp(node->key, key) == 0) {
      if (prev) {
        prev->next = node->next;
      } else {
        dictionary->buckets[index] = node->next;
      }
      void *value = node->value;
      free(node->key);
      free(node);
      dictionary->size--;
      *err = false;
      return value;
    }
    prev = node;
    node = node->next;
  }
  *err = true;
  return NULL;
};

bool dictionary_contains(dictionary_t *dictionary, const char *key) {
  /* Indica si hay un valor asociado a la clave indicada. O(1).
 * Pre-condiciones:
 * - El diccionario existe
 * - La clave tiene largo mayor a cero
 * Post-condiciones:
 * - Retorna true si la clave está presente en el diccionario
 * - Retorna false de otro modo
 */
    unsigned long index = hash(key, dictionary->capacity);
    struct entry_node *current = dictionary->buckets[index];
    
    while (current) {
        if (strcmp(current->key, key) == 0)
            return true;
        current = current->next;
    }
    return false;
};

size_t dictionary_size(dictionary_t *dictionary) { 
  /* Indica la cantidad de elementos guardados en el diccionario. O(1).
 * Pre-condiciones:
 * - El diccionario existe
 */
return dictionary->size; };

void dictionary_destroy(dictionary_t *dictionary){
  /* Destruye el diccionario y los valores asociados a todas las claves presentes.
 * Pre-condiciones:
 * - El diccionario existe
 */
  if (!dictionary) return;
  for (size_t i = 0; i < dictionary->capacity; i++) {
    struct entry_node *node = dictionary->buckets[i];
    while (node) {
      struct entry_node *next = node->next;
      entry_node_destroy(node, dictionary->destroy);
      node = next;
    }
  }
  free(dictionary->buckets);
  free(dictionary);
};


