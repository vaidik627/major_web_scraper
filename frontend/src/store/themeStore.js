import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useThemeStore = create(
  persist(
    (set, get) => ({
      isDark: false,
      
      toggleTheme: () => {
        set((state) => ({ isDark: !state.isDark }));
      },
      
      setTheme: (isDark) => {
        set({ isDark });
      },
      
      initializeTheme: () => {
        const { isDark } = get();
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // If no preference is stored, use system preference
        if (isDark === undefined) {
          set({ isDark: systemPrefersDark });
        }
      },
    }),
    {
      name: 'theme-storage',
      partialize: (state) => ({ isDark: state.isDark }),
    }
  )
);