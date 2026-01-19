import { createContext, useContext, useState, useEffect } from 'react'

const LanguageContext = createContext()

export function useLanguage() {
    return useContext(LanguageContext)
}

export const LANGUAGES = [
    { code: 'en', name: 'English', native: 'English' },
    { code: 'kn', name: 'Kannada', native: 'ಕನ್ನಡ' },
    { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
    { code: 'te', name: 'Telugu', native: 'తెలుగు' },
    { code: 'ta', name: 'Tamil', native: 'தமிழ்' }
]

export function LanguageProvider({ children }) {
    const [language, setLanguage] = useState('English')

    const value = {
        language,
        setLanguage,
        LANGUAGES
    }

    return (
        <LanguageContext.Provider value={value}>
            {children}
        </LanguageContext.Provider>
    )
}
