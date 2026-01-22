
import ipywidgets as widgets
from IPython.display import display, HTML

class UIStyles:
    """
    Manages global CSS styles, animations, and micro-interactions for the Jupyter UI.
    Injects modern CSS variables, keyframes, and component styles.
    """
    
    @staticmethod
    def get_css():
        return """
        <style>
            /* =========================================
               🚀 ANIMATION VARIABLES & KEYFRAMES
               ========================================= */
            :root {
                --primary-color: #3b82f6;
                --primary-hover: #2563eb;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --bg-glass: rgba(255, 255, 255, 0.95);
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-glow: 0 0 15px rgba(59, 130, 246, 0.5);
            }

            /* Pulse Animation (for active states) */
            @keyframes pulse-soft {
                0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
                100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
            }
            
            @keyframes pulse-green {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }

            /* Fade In Animation */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Gradient Background Animation */
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            /* =========================================
               ✨ COMPONENT STYLES
               ========================================= */

            /* 1. PRIMARY BUTTONS (Run, Calculate) */
            .jupyter-widgets.widget-button.premium-btn {
                background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
                color: white !important;
                font-weight: 600;
                border-radius: 8px;
                border: none;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: var(--shadow-md);
                letter-spacing: 0.5px;
            }
            .jupyter-widgets.widget-button.premium-btn:hover {
                transform: translateY(-2px) scale(1.02);
                box-shadow: var(--shadow-lg), var(--shadow-glow);
                background: linear-gradient(135deg, var(--primary-hover), var(--primary-color));
            }
            .jupyter-widgets.widget-button.premium-btn:active {
                transform: translateY(1px);
            }

            /* 2. SUCCESS BUTTONS (Export, Save) */
            .jupyter-widgets.widget-button.success-btn {
                background: linear-gradient(135deg, #10b981, #059669);
                color: white !important;
                border-radius: 8px;
                transition: all 0.2s ease;
            }
            .jupyter-widgets.widget-button.success-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            }

            /* 3. CARDS & CONTAINERS */
            .glass-card {
                background: var(--bg-glass);
                border-radius: 12px;
                padding: 15px;
                border: 1px solid rgba(255,255,255,0.2);
                box-shadow: var(--shadow-sm);
                transition: box-shadow 0.3s ease;
                animation: fadeIn 0.5s ease-out;
            }
            .glass-card:hover {
                box-shadow: var(--shadow-md);
            }

            /* 4. PROGRESS BAR ANIMATION */
            .jupyter-widgets.widget-progress .progress-bar {
                background-image: linear-gradient(
                    45deg, 
                    rgba(255, 255, 255, 0.15) 25%, 
                    transparent 25%, 
                    transparent 50%, 
                    rgba(255, 255, 255, 0.15) 50%, 
                    rgba(255, 255, 255, 0.15) 75%, 
                    transparent 75%, 
                    transparent
                );
                background-size: 1rem 1rem;
                animation: progress-stripe 1s linear infinite;
                border-radius: 8px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            }
            @keyframes progress-stripe {
                from { background-position: 1rem 0; }
                to { background-position: 0 0; }
            }

            /* 5. STATUS INDICATORS */
            .status-running {
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: var(--primary-color);
                border-radius: 50%;
                animation: pulse-soft 2s infinite;
                margin-right: 8px;
            }
            .status-success {
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: var(--success-color);
                border-radius: 50%;
                box-shadow: 0 0 5px var(--success-color);
                margin-right: 8px;
            }

            /* 6. DROPDOWNS & INPUTS */
            .jupyter-widgets.widget-dropdown select, 
            .jupyter-widgets.widget-text input {
                border-radius: 6px;
                border: 1px solid #e5e7eb;
                padding: 6px 10px;
                transition: all 0.2s;
            }
            .jupyter-widgets.widget-dropdown select:focus, 
            .jupyter-widgets.widget-text input:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                outline: none;
            }

            /* 7. TOAST NOTIFICATIONS */
            .toast-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                gap: 10px;
                pointer-events: none;
            }
            .toast-message {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-left: 5px solid var(--primary-color);
                padding: 15px 20px;
                border-radius: 4px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
                animation: slideInRight 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                display: flex;
                align-items: center;
                min-width: 250px;
                pointer-events: auto;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            .toast-success { border-left-color: var(--success-color); }
            .toast-error { border-left-color: var(--danger-color); }
            .toast-warning { border-left-color: var(--warning-color); }
            
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(100%); }
                to { opacity: 1; transform: translateX(0); }
            }
            @keyframes fadeOutRight {
                from { opacity: 1; transform: translateX(0); }
                to { opacity: 0; transform: translateX(100%); }
            }

            /* 8. SKELETON LOADING (Shimmer) */
            .skeleton {
                background: #f0f0f0;
                background-image: linear-gradient(
                    90deg, 
                    #f0f0f0 0px, 
                    #f8f8f8 40px, 
                    #f0f0f0 80px
                );
                background-size: 300px 100%;
                animation: shimmer 1.5s infinite linear;
                border-radius: 4px;
            }
            @keyframes shimmer {
                0% { background-position: -300px 0; }
                100% { background-position: 300px 0; }
            }

            /* 9. CHECKMARK ANIMATION */
            .checkmark-circle {
                width: 20px; height: 20px; position: relative; display: inline-block; vertical-align: middle; margin-right: 8px;
            }
            .checkmark-stem {
                position: absolute; width: 2px; height: 10px; background-color: var(--success-color); left: 10px; top: 2px; transform: rotate(45deg);
                animation: stem-grow 0.2s 0.2s ease-out forwards; opacity: 0;
            }
            .checkmark-kick {
                position: absolute; width: 6px; height: 2px; background-color: var(--success-color); left: 5px; top: 10px; transform: rotate(45deg);
                animation: kick-grow 0.2s ease-out forwards; opacity: 0;
            }
            @keyframes stem-grow { from { height: 0; opacity: 0; } to { height: 10px; opacity: 1; } }
            @keyframes kick-grow { from { width: 0; opacity: 0; } to { width: 6px; opacity: 1; } }
            
        </style>
        
        <- Include validation script Inject Toast Container (JS/HTML hack) -->
        <div id="jupyter-toast-container" class="toast-container"></div>

        """

    @staticmethod
    def inject():
        """Injects the CSS styles into the current notebook."""
        display(HTML(UIStyles.get_css()))

    @staticmethod
    def apply_premium_style(widget_instance):
        """Applies premium styling class to a button widget."""
        widget_instance.add_class('premium-btn')
        
    @staticmethod
    def apply_success_style(widget_instance):
        """Applies success styling class to a button widget."""
        widget_instance.add_class('success-btn')
        

    @staticmethod
    def show_toast(title, message, type='info'):
        """
        Triggers a toast notification using Javascript execution.
        Types: 'info', 'success', 'error', 'warning'
        """
        import uuid
        toast_id = f"toast-{uuid.uuid4()}"
        icon = "ℹ️"
        if type == 'success': icon = "✅"
        if type == 'error': icon = "❌"
        if type == 'warning': icon = "⚠️"
        
        js = f"""
        <script>
        (function() {{
            const container = document.getElementById('jupyter-toast-container');
            if (!container) {{
                const newContainer = document.createElement('div');
                newContainer.id = 'jupyter-toast-container';
                newContainer.className = 'toast-container';
                document.body.appendChild(newContainer);
            }}
            
            const toast = document.createElement('div');
            toast.className = 'toast-message toast-{type}';
            toast.id = '{toast_id}';
            toast.innerHTML = `
                <div style="margin-right: 12px; font-size: 18px;">{icon}</div>
                <div>
                    <div style="font-weight: 700; margin-bottom: 2px;">{title}</div>
                    <div style="color: #666;">{message}</div>
                </div>
            `;
            
            document.getElementById('jupyter-toast-container').appendChild(toast);
            
            // Remove after 4 seconds
            setTimeout(() => {{
                toast.style.animation = 'fadeOutRight 0.5s forwards';
                setTimeout(() => {{ toast.remove(); }}, 500);
            }}, 4000);
        }})();
        </script>
        """
        display(HTML(js))

    @staticmethod
    def get_skeleton_html(height='20px', width='100%'):
        """Returns HTML for a skeleton loader div."""
        return f"<div class='skeleton' style='height: {height}; width: {width};'></div>"
        
    @staticmethod
    def apply_card_style(widget_instance):
        """Applies card/container styling."""
        widget_instance.add_class('glass-card')

