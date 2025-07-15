// Enhanced Medical Condition Predictor JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all enhancements
    initializeAnimations();
    initializeCounters();
    initializeFormEnhancements();
    initializeTooltips();
    initializeScrollEffects();
    initializeDrugCards();
});

// 1. Initialize Page Animations
function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.feature-card, .project-info-card, .conditions-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.8s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 200);
    });
    
    // Add stagger animation to statistics
    const statItems = document.querySelectorAll('.stat-item');
    statItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            item.style.transition = 'all 0.6s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, index * 100 + 500);
    });
}

// 2. Animated Counters for Statistics
function initializeCounters() {
    const counters = document.querySelectorAll('.stat-number');
    
    counters.forEach(counter => {
        const target = counter.textContent;
        const isNumber = /^\d+/.test(target);
        
        if (isNumber) {
            const finalNumber = parseInt(target.replace(/\D/g, ''));
            const suffix = target.replace(/^\d+/, '');
            
            animateCounter(counter, finalNumber, suffix, 2000);
        }
    });
}

function animateCounter(element, target, suffix, duration) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            element.textContent = target + suffix;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start) + suffix;
        }
    }, 16);
}

// 3. Enhanced Form Interactions
function initializeFormEnhancements() {
    // Enhanced textarea behavior
    const textarea = document.querySelector('#rawtext');
    if (textarea) {
        // Auto-resize textarea
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Character counter
        const maxLength = 1000;
        const counter = document.createElement('div');
        counter.className = 'character-counter';
        counter.style.cssText = `
            text-align: right;
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.5rem;
        `;
        
        textarea.addEventListener('input', function() {
            const remaining = maxLength - this.value.length;
            counter.textContent = `${this.value.length}/${maxLength} characters`;
            
            if (remaining < 100) {
                counter.style.color = '#e74c3c';
            } else if (remaining < 200) {
                counter.style.color = '#f39c12';
            } else {
                counter.style.color = '#6c757d';
            }
        });
        
        textarea.parentNode.appendChild(counter);
        
        // Add typing animation effect
        textarea.addEventListener('input', function() {
            this.style.borderColor = '#667eea';
            this.style.boxShadow = '0 0 0 0.2rem rgba(102, 126, 234, 0.25)';
            
            setTimeout(() => {
                this.style.borderColor = 'rgba(102, 126, 234, 0.2)';
                this.style.boxShadow = 'none';
            }, 300);
        });
    }
    
    // Enhanced button interactions
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
        
        button.addEventListener('click', function() {
            // Create ripple effect
            const ripple = document.createElement('span');
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

// 4. Initialize Tooltips and Popovers
function initializeTooltips() {
    // Add tooltips to stat items
    const statItems = document.querySelectorAll('.stat-item');
    const tooltips = [
        'Real patient reviews and experiences',
        'Supported medical conditions',
        'Available drug recommendations',
        'Powered by advanced AI technology'
    ];
    
    statItems.forEach((item, index) => {
        if (tooltips[index]) {
            item.setAttribute('data-bs-toggle', 'tooltip');
            item.setAttribute('data-bs-placement', 'top');
            item.setAttribute('title', tooltips[index]);
        }
    });
    
    // Initialize Bootstrap tooltips
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// 5. Scroll-based Animations
function initializeScrollEffects() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements for scroll animations
    const elementsToObserve = document.querySelectorAll('.feature-card, .condition-item, .info-row');
    elementsToObserve.forEach(el => observer.observe(el));
}

// 6. Enhanced Drug Card Interactions
function initializeDrugCards() {
    const drugCards = document.querySelectorAll('.drug-card');
    
    drugCards.forEach(card => {
        // Add loading state when clicked
        card.addEventListener('click', function() {
            const originalContent = this.innerHTML;
            
            // Show loading state
            this.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading reviews...</p>
                </div>
            `;
            
            // Restore content after a short delay (in real app, this would be when data loads)
            setTimeout(() => {
                this.innerHTML = originalContent;
            }, 1500);
        });
        
        // Add hover effect for drug stats
        const statBadges = card.querySelectorAll('.stat-badge');
        statBadges.forEach(badge => {
            badge.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.1)';
                this.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.4)';
            });
            
            badge.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = 'none';
            });
        });
    });
}

// 7. Enhanced Confidence Bar Animation
function animateConfidenceBar(percentage) {
    const confidenceBar = document.querySelector('.confidence-fill');
    if (confidenceBar) {
        confidenceBar.style.width = '0%';
        
        setTimeout(() => {
            confidenceBar.style.width = percentage + '%';
            
            // Add pulsing effect for high confidence
            if (percentage > 80) {
                confidenceBar.style.animation = 'pulse 2s infinite';
            }
        }, 500);
    }
}

// 8. Form Validation Enhancements
function enhanceFormValidation() {
    const form = document.querySelector('#predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            const textarea = document.querySelector('#rawtext');
            const text = textarea.value.trim();
            
            // Clear previous validation messages
            const existingError = document.querySelector('.validation-error');
            if (existingError) existingError.remove();
            
            if (text.length < 10) {
                e.preventDefault();
                showValidationError(textarea, 'Please enter at least 10 characters for better prediction accuracy.');
                return false;
            }
            
            if (text.length > 1000) {
                e.preventDefault();
                showValidationError(textarea, 'Please limit your description to 1000 characters.');
                return false;
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                submitBtn.disabled = true;
            }
        });
    }
}

function showValidationError(element, message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'validation-error alert alert-error mt-2';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${message}`;
    
    element.parentNode.appendChild(errorDiv);
    
    // Shake animation for the input
    element.style.animation = 'shake 0.5s ease-in-out';
    setTimeout(() => {
        element.style.animation = '';
    }, 500);
    
    // Focus back to the input
    element.focus();
}

// 9. Real-time Search Enhancement (for drug cards)
function initializeSearch() {
    const searchInput = document.querySelector('#drugSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const drugCards = document.querySelectorAll('.drug-card');
            
            drugCards.forEach(card => {
                const drugName = card.querySelector('.drug-name').textContent.toLowerCase();
                if (drugName.includes(searchTerm)) {
                    card.style.display = 'block';
                    card.style.opacity = '1';
                } else {
                    card.style.opacity = '0.3';
                    setTimeout(() => {
                        if (!drugName.includes(searchInput.value.toLowerCase())) {
                            card.style.display = 'none';
                        }
                    }, 300);
                }
            });
        });
    }
}

// 10. Enhanced Modal Interactions
function enhanceModalInteractions() {
    // Smooth modal transitions
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.addEventListener('show.bs.modal', function() {
            this.style.display = 'block';
            this.style.opacity = '0';
            setTimeout(() => {
                this.style.opacity = '1';
            }, 10);
        });
        
        modal.addEventListener('hide.bs.modal', function() {
            this.style.opacity = '0';
            setTimeout(() => {
                this.style.display = 'none';
            }, 300);
        });
    });
    
    // Add close button animation
    const closeButtons = document.querySelectorAll('.modal .btn-close');
    closeButtons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'rotate(90deg) scale(1.1)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'rotate(0deg) scale(1)';
        });
    });
}

// 11. Performance Optimizations
function initializePerformanceOptimizations() {
    // Lazy load images if any
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                observer.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Debounced resize handler
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            // Recalculate any responsive elements
            adjustResponsiveElements();
        }, 250);
    });
}

function adjustResponsiveElements() {
    // Adjust card heights on mobile
    if (window.innerWidth < 768) {
        const featureCards = document.querySelectorAll('.feature-card');
        featureCards.forEach(card => {
            card.style.height = 'auto';
            card.style.minHeight = '250px';
        });
    }
}

// 12. CSS Animation Helpers
const animationHelpers = {
    // Fade in animation
    fadeIn: (element, duration = 500) => {
        element.style.opacity = '0';
        element.style.transition = `opacity ${duration}ms ease`;
        
        setTimeout(() => {
            element.style.opacity = '1';
        }, 10);
    },
    
    // Slide up animation
    slideUp: (element, duration = 500) => {
        element.style.transform = 'translateY(20px)';
        element.style.opacity = '0';
        element.style.transition = `all ${duration}ms ease`;
        
        setTimeout(() => {
            element.style.transform = 'translateY(0)';
            element.style.opacity = '1';
        }, 10);
    },
    
    // Scale animation
    scale: (element, scale = 1.05, duration = 200) => {
        element.style.transition = `transform ${duration}ms ease`;
        element.style.transform = `scale(${scale})`;
        
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, duration);
    }
};

// 13. Add CSS for shake animation
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .character-counter {
        transition: color 0.3s ease;
    }
    
    .validation-error {
        animation: slideDown 0.3s ease;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// 14. Initialize all enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeAnimations();
    initializeCounters();
    initializeFormEnhancements();
    initializeTooltips();
    initializeScrollEffects();
    initializeDrugCards();
    enhanceFormValidation();
    initializeSearch();
    enhanceModalInteractions();
    initializePerformanceOptimizations();
    
    // Add page load animation
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease';
        document.body.style.opacity = '1';
    }, 100);
});

// 15. Export functions for use in other scripts
window.medicalPredictorEnhancements = {
    animateConfidenceBar,
    animationHelpers,
    showValidationError
};