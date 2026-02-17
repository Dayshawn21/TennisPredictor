const baseUrl = (import.meta as any).env?.REACT_APP_API_BASE_URL || ''

export const getData = (userId: number): string => {
    return baseUrl + '/data/' + userId
}
