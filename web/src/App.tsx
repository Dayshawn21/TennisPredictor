import React from 'react'
import { Provider } from 'react-redux'
import { PersistGate } from 'redux-persist/integration/react'
import { ConfigProvider, theme } from 'antd'
import RootComponent from './RootComponent'
import { persistor, store } from './store/reducers/store'
import 'antd/dist/reset.css'

const App: React.FC = () => {
    return (
        <Provider store={store}>
            <PersistGate loading={null} persistor={persistor}>
                <ConfigProvider
                    theme={{
                        algorithm: theme.darkAlgorithm,
                        cssVar: true // ✅ IMPORTANT if you want --ant-color-text to exist
                    }}
                >
                    <div className="min-h-screen bg-gray-900 text-gray-100">
                        <RootComponent />
                    </div>
                </ConfigProvider>
            </PersistGate>
        </Provider>
    )
}

export default App
