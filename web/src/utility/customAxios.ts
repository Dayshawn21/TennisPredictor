import axios from 'axios'

type JsonLike = null | boolean | number | string | JsonLike[] | { [key: string]: JsonLike }

const CustomAxios = axios.create({
    baseURL: import.meta.env?.REACT_APP_API_BASE_URL || undefined
})

const toCamelCase = (object: JsonLike): JsonLike => {
    let transformedObject = object
    if (typeof object === 'object' && object !== null) {
        if (Array.isArray(object)) {
            transformedObject = object.map(toCamelCase)
        } else {
            transformedObject = {}
            for (const key in object) {
                if (Object.prototype.hasOwnProperty.call(object, key) && object[key] !== undefined) {
                    const newKey = key.replace(/(_\w)|(-\w)/g, (k) => k[1].toUpperCase())
                    ;(transformedObject as Record<string, JsonLike>)[newKey] = toCamelCase(object[key])
                }
            }
        }
    }
    return transformedObject
}

export const toSnakeCase = (object: JsonLike): JsonLike => {
    let transformedObject = object
    if (typeof object === 'object' && object !== null) {
        if (Array.isArray(object)) {
            transformedObject = object.map(toSnakeCase)
        } else {
            transformedObject = {}
            for (const key in object) {
                if (Object.prototype.hasOwnProperty.call(object, key) && object[key] !== undefined) {
                    const newKey = key
                        .replace(/\.?([A-Z]+)/g, function (_, y) {
                            return '_' + y.toLowerCase()
                        })
                        .replace(/^_/, '')
                    ;(transformedObject as Record<string, JsonLike>)[newKey] = toSnakeCase(object[key])
                }
            }
        }
    }
    return transformedObject
}

// Compatibility alias for existing imports.
export const toSnackCase = toSnakeCase

CustomAxios.interceptors.response.use(
    (response) => {
        response.data = toCamelCase(response.data as JsonLike)
        return response
    },
    (error) => {
        return Promise.reject(error)
    }
)

CustomAxios.interceptors.request.use(
    (config) => {
        if (config.data !== undefined) config.data = toSnakeCase(config.data as JsonLike)
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)

export default CustomAxios
