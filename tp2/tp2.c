#include "tp2.h"
#include <stdlib.h>
#include <stdbool.h>

struct node;
typedef struct node node_t;

struct node {
    void* value;
    node_t* next;
    node_t* prev;
};

struct list {
    node_t* head;
    node_t* tail;
    size_t size;
};

struct list_iter {
    list_t* list;
    node_t* curr;
};

list_t *list_new() {
    list_t *list = malloc(sizeof(list_t));
    if (!list) {return NULL;}
    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    return list;
}

size_t list_length(const list_t *list){
    return list->size;
}

bool list_is_empty(const list_t *list){
    return list->size == 0;
}

bool list_insert_head(list_t *list, void *value){
    node_t *node = malloc(sizeof(node_t));
    if(!node){
        return false;
    }
    node->value = value;
    node->next = list->head;
    if(!(list->head)){
        list->head = node;
        list->tail = node;
        list->size++;
        return true;
    }
    node->prev = NULL;
    list->head->prev = node;
    list->head = node;
    list->size++;
    return true;
}

bool list_insert_tail(list_t *list, void *value){
    node_t *node = malloc(sizeof(node_t));
    if(!node){
        return false;
    }
    node->value = value;
    node->next = NULL;
    node->prev = list->tail;
    if(list->tail){
        list->tail->next = node;
    }
    list->tail = node;
    if(!(list->head)){
        list->head = node;
    }
    list->size++;
    return true;
}

void *list_peek_head(const list_t *list){
    if(list_is_empty(list)){return NULL;}
    return list->head->value;
}

void *list_peek_tail(const list_t *list){
    if(list_is_empty(list)){return NULL;}
    return list->tail->value;
}

void *list_pop_head(list_t *list){
    if(!(list->head)){return NULL;}
    if(list->head == list->tail){
        void *value = list->head->value;
        free(list->head);
        list->head = NULL;
        list->tail = NULL;
        list->size--;
        return value;
    }
    node_t *curr = list->head;
    void *value = curr->value;
    list->head = curr->next;
    list->head->prev = NULL;
    free(curr);
    list->size--;
    return value;
}

void *list_pop_tail(list_t *list){
    if(!(list->head)){return NULL;}
    if(list->head == list->tail){
        void *value = list->head->value;
        free(list->head);
        list->head = NULL;
        list->tail = NULL;
        list->size--;
        return value;
    }
    node_t *curr = list->tail;
    void *value = curr->value;
    list->tail = curr->prev;
    list->tail->next = NULL;
    free(curr);
    list->size--;
    return value;
}

void list_destroy(list_t *list, void destroy_value(void *)){
    if(!list){return;}
    node_t *curr = list->head;
    while(curr){
        node_t *next = curr->next;
        if(destroy_value != NULL){
            destroy_value(curr->value);
        }
        free(curr);
        curr = next;
    }
    free(list);
}

list_iter_t *list_iter_create_head(list_t *list){
    if(!list){return NULL;}
    list_iter_t *list_iter = malloc(sizeof(list_iter_t));
    if(!(list_iter)){return NULL;}
    list_iter->list = list;
    list_iter->curr = list->head;
    return list_iter; 
}

list_iter_t *list_iter_create_tail(list_t *list){
    if(!list){return NULL;}
    list_iter_t *list_iter = malloc(sizeof(list_iter_t));
    if(!(list_iter)){return NULL;}
    list_iter->list = list;
    list_iter->curr = list->tail;
    return list_iter; 
}

bool list_iter_forward(list_iter_t *iter){
    if(!iter || list_is_empty(iter->list) || list_iter_at_last(iter)){return false;}
    iter->curr = iter->curr->next;
    return true;
}

bool list_iter_backward(list_iter_t *iter){
    if(!iter || list_is_empty(iter->list) || list_iter_at_first(iter)){return false;}
    iter->curr = iter->curr->prev;
    return true;
}

void *list_iter_peek_current(const list_iter_t *iter){
    if(list_is_empty(iter->list)){return NULL;}
    return iter->curr->value;
}

bool list_iter_at_last(const list_iter_t *iter){
    return iter->curr == iter->list->tail;
}

bool list_iter_at_first(const list_iter_t *iter){
    return iter->curr == iter->list->head;
}

void list_iter_destroy(list_iter_t *iter){
    free(iter);
}
bool list_iter_insert_after(list_iter_t *iter, void *value) {
    if (!iter || !iter->list) return false;

    if (value == NULL) return false;

    if (list_is_empty(iter->list) || list_iter_at_last(iter)) {
        list_insert_tail(iter->list, value);
        iter->curr = iter->list->tail;
        return true;
    }

    node_t *node = malloc(sizeof(node_t));
    if (!node) return false;

    node->value = value;
    node->next = iter->curr->next;
    node->prev = iter->curr;

    iter->curr->next = node;
    node->next->prev = node;

    iter->list->size++;

    return true;
}

bool list_iter_insert_before(list_iter_t *iter, void *value) {
    if (!iter || !iter->list) return false;

    if (value == NULL) return false;

    if (list_is_empty(iter->list) || list_iter_at_first(iter)) {
        list_insert_head(iter->list, value);
        iter->curr = iter->list->head;
        return true;
    }

    node_t *node = malloc(sizeof(node_t));
    if (!node) return false;

    node->value = value;
    node->prev = iter->curr->prev;
    node->next = iter->curr;

    iter->curr->prev = node;
    node->prev->next = node;

    iter->list->size++;

    return true;
}

void *list_iter_delete(list_iter_t *iter) {
    if (list_is_empty(iter->list)) {
        return NULL;
    }

    node_t *node = iter->curr;
    void *value = node->value;

    if (list_iter_at_last(iter)) {
        iter->list->tail = node->prev;
        if (node->prev) {
            node->prev->next = NULL;
        } else {
            iter->list->head = NULL;
            iter->list->tail = NULL;
        }
    } else if (list_iter_at_first(iter)) {
        iter->list->head = node->next;
        node->next->prev = NULL;
    } else {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    iter->curr = (node->next) ? node->next : node->prev;

    free(node);
    iter->list->size--;
    return value;
}